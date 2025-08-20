import time
import re
import sys
import urllib.parse
from typing import List, Dict, Any, Optional, Tuple
import os
import pandas as pd
import numpy as np
import random
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import osmnx as ox
import networkx as nx
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

BUDGET_TO_INR = {"Low": 1000, "Mid": 3000, "Medium": 3000, "High": 5000}
OSM_NETWORK_TYPE = "drive" 
OSM_GRAPH_DIST = 10000         
GEOCODE_SLEEP = 1.0           
NOMINATIM_USER_AGENT = "kolkata_itinerary_planner_v1"


class KolkataItineraryGenerator:
    def __init__(self, hotels_csv: str, hotel_recommendation_csv:str, tourist_csv: str, shopping_csv: str, nearby_hotels_csv: Optional[str] = None):
        try:
            self.hotels_df = pd.read_csv(hotels_csv)
            self.hotelsRec_df=pd.read_csv(hotel_recommendation_csv)
            self.tourist_df = pd.read_csv(tourist_csv)
            self.shopping_df = pd.read_csv(shopping_csv)
            self.hotels_df.columns = self.hotels_df.columns.str.strip()
            self.tourist_df.columns = self.tourist_df.columns.str.strip()
            self.shopping_df.columns = self.shopping_df.columns.str.strip()
        except FileNotFoundError as e:
            print(f"Error: The file {e.filename} was not found.")
            sys.exit(1)
        except Exception as e:
            print(f"An error occurred while loading CSVs: {e}")
            sys.exit(1)

        self.nearby_hotels_df = None
        if nearby_hotels_csv:
            try:
                nh = pd.read_csv(nearby_hotels_csv)
                nh.columns = nh.columns.str.strip()
                expected = ["Hotel", "Latitude", "Longitude", "Budget", "Nearby"]
                self.nearby_hotels_df = nh.copy()
            except Exception as e:
                print(f"Warning: Could not load nearby_hotels_csv ({nearby_hotels_csv}). Error: {e}")
                self.nearby_hotels_df = None

        # required column names from your original code
        self.tourist_name_col = self._first_col(self.tourist_df, ["Name", "Spot", "Place"], required=True)
        self.shopping_name_col = self._first_col(self.shopping_df, ["Name", "Mall", "Market"], required=True)
        self.hotels_name_col = self._first_col(self.hotels_df, ["Hotel", "Hotel Name"], required=True)
        self.hotels_budget_col = self._first_col(self.hotels_df, ["Budget", "Price Category"], required=True)

        # ML placeholders (unchanged)
        self.ml_model = None
        self.scaler = None

        # OSMnx / geocoding caches
        self._osm_graph: Optional[nx.MultiDiGraph] = None
        self._osm_hotel_node: Optional[int] = None
        self._hotel_coord: Optional[Tuple[float, float]] = None
        self._geolocator: Optional[Nominatim] = None

    def _norm(self, s: Any) -> str:
        if pd.isna(s):
            return ""
        return str(s).replace('’', "'").strip().casefold()

    def _first_col(self, df: pd.DataFrame, candidates: List[str], required: bool = False) -> Optional[str]:
        for c in candidates:
            if c in df.columns:
                return c
        if required:
            raise KeyError(f"Missing any of required columns: {candidates}")
        return None

    def _safe_info(self, df: pd.DataFrame, name: str, source_type: str) -> (str, str):
        name_col = self._first_col(df, ["Name", "Spot", "Place", "Mall", "Market"])
        if not name_col:
            return "", ""

        normalized_name = self._norm(name)
        cand = df[df[name_col].map(self._norm) == normalized_name]
        if cand.empty:
            return "", ""

        desc = ""
        acts = ""

        desc_cols = ["Description", "Details", "Info", "About"]
        desc_col = self._first_col(df, desc_cols)
        if desc_col and pd.notna(cand[desc_col].iloc[0]) and str(cand[desc_col].iloc[0]).strip():
            desc = str(cand[desc_col].iloc[0]).strip()

        acts_cols = ["Top Activities", "Activities", "Things To Do", "What To Do"]
        acts_col = self._first_col(df, acts_cols)
        if acts_col and pd.notna(cand[acts_col].iloc[0]) and str(cand[acts_col].iloc[0]).strip():
            acts = str(cand[acts_col].iloc[0]).strip()

        return desc, acts

    def _parse_spots(self, spots_cell: Any) -> List[Dict[str, Any]]:
        if pd.isna(spots_cell) or self._norm(spots_cell) == "":
            return []
        text = str(spots_cell)
        parts = re.split(r"→|->|;|\|", text)
        results = []
        for raw in parts:
            s = raw.strip()
            if not s:
                continue
            m = re.search(r"\(([^)]*)\)$", s)
            meta = m.group(1).strip() if m else ""
            name = s[: m.start()].strip() if m else s
            kind = "visit"
            minutes = None
            if meta:
                low = meta.lower()
                if "lunch" in low:
                    kind = "lunch"
                elif "dinner" in low:
                    kind = "dinner"
                elif "shopping" in low:
                    kind = "shopping"
                m_minutes = re.search(r"(\d+)\s*m", low)
                minutes = int(m_minutes.group(1)) if m_minutes else None
            results.append({"name": name, "kind": kind, "minutes": minutes})
        return results

    def _parse_meal_alloc(self, cell: Any) -> Dict[str, int]:
        default = {"Breakfast": 45, "Lunch": 75, "Snacks": 30, "Dinner": 75}
        if pd.isna(cell) or self._norm(cell) == "":
            return default
        text = str(cell)
        out: Dict[str, int] = {}
        for part in text.split(","):
            kv = part.strip()
            m = re.match(r"([A-Za-z]+)\s+(\d+)\s*m", kv)
            if m:
                out[m.group(1).title()] = int(m.group(2))
        for k, v in default.items():
            out.setdefault(k, v)
        return out

    def _get_transportation_tips(self, budget_value: int, group_type: str, transportation_mode: str) -> str:
        tips = "\n🚕 Transportation Tips:\n"
        norm_mode = self._norm(transportation_mode)
        budget_category = self._map_budget_to_category(budget_value)

        if norm_mode == "public transport":
            tips += "  - Use the Kolkata Metro for fast and cheap travel.\n"
        elif norm_mode == "ride-hailing":
            tips += "  - Uber and Ola are convenient and available across the city.\n"
        elif norm_mode == "private cab":
            tips += "  - Hire a private cab with driver for comfort and flexibility.\n"
        else:
            if budget_category == "Low":
                tips += "  Best option: public transport (Metro, buses, trams).\n"
            elif budget_category == "Medium":
                tips += "  Mix of Metro and Uber/Ola is ideal.\n"
            else:
                tips += "  Focus on comfort with cabs or app-based taxis.\n"

        if "family" in self._norm(group_type) or "couple" in self._norm(group_type):
            tips += "\n  Tip: Private cabs recommended for comfort."

        return tips

    def _map_budget_to_category(self, budget_value: int) -> str:
        if budget_value <= 1000:
            return "Low"
        elif 1000 < budget_value <= 3000:
            return "Medium"
        else:
            return "High"

    def _generate_maps_url(self, locations: List[str]) -> str:
        if not locations:
            return "(No route available)"
        encoded_places = [urllib.parse.quote_plus(loc) for loc in locations]
        return "https://www.google.com/maps/dir/" + "/".join(encoded_places)

    def train_ml_models(self):
        X = np.array([
            [1000, 4.5], [2500, 4.0], [500, 3.8], [5000, 4.8],
            [1500, 4.2], [3500, 4.6], [800, 3.5], [2000, 4.1]
        ])
        y = np.array([4.2, 4.0, 3.9, 4.7, 4.3, 4.5, 3.6, 4.1])
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        self.ml_model = LinearRegression()
        self.ml_model.fit(X_scaled, y)

    def _arrival_zone(self, arrival_area: str) -> str:
        """Map user-provided arrival area to a normalized zone key."""
        aa = self._norm(arrival_area)
        if any(k in aa for k in ["airport", "dum dum", "dumdum", "flight", "nscbi", "netaji"]):
            return "airport"
        if "howrah" in aa:
            return "howrah"
        if "sealdah" in aa:
            return "sealdah"
        if any(k in aa for k in ["esplanade", "bus", "dharmatala", "dharamtala"]):
            return "esplanade"
        # fallback: use raw text as zone-ish
        return aa or "airport"

    def _arrival_keywords(self, arrival_area: str) -> List[str]:
        """Robust synonyms per zone."""
        zone = self._arrival_zone(arrival_area)
        if zone == "airport":
            return [
                "airport", "nscb", "nscbi", "netaji subhash", "netaji subhas", "netaji subhash chandra bose",
                "netaji subhas chandra bose", "kolkata airport", "dum dum", "dumdum", "ns c b i"
            ]
        if zone == "howrah":
            return ["howrah", "howrah jn", "howrah railway"]
        if zone == "sealdah":
            return ["sealdah", "sealdah jn", "sealdah railway"]
        if zone == "esplanade":
            return ["esplanade", "bus stand", "dharmatala", "dharamtala"]
        # fallback: try the raw area word itself
        return [zone]

    def _budget_ok(self, budget_inr: int, hotel_budget_label: str) -> bool:
        price_est = BUDGET_TO_INR.get(hotel_budget_label, 2500)
        return price_est <= (budget_inr + 1000)

    def _maps_place_url_from_coords(self, name: str, lat: Any, lon: Any) -> str:
        try:
            return f"https://www.google.com/maps/search/?api=1&query={urllib.parse.quote_plus(name)}&query_place_id={lat},{lon}"
        except Exception:
            return f"https://www.google.com/maps/search/{urllib.parse.quote_plus(name)}"

    def _choose_nearby_hotel(self, daily_budget: int, arrival_area: str) -> Optional[str]:
        if self.nearby_hotels_df is None or self.nearby_hotels_df.empty:
            return None

        df = self.nearby_hotels_df.copy()
        if "Budget" in df.columns:
            df = df[df["Budget"].apply(lambda x: self._budget_ok(daily_budget, str(x)))]
        # score nearby text
        keys = self._arrival_keywords(arrival_area)
        def match_score(text: str) -> int:
            t = self._norm(str(text))
            return sum(1 for k in keys if k in t)
        if "Nearby" in df.columns:
            df["__near_score"] = df["Nearby"].astype(str).apply(match_score)
        else:
            df["__near_score"] = 0
        def budget_rank(lbl: str) -> int:
            lbln = self._norm(str(lbl))
            if lbln in ("low",): return 0
            if lbln in ("mid", "medium"): return 1
            if lbln in ("high",): return 2
            return 3
        if "Budget" in df.columns:
            df["__budget_rank"] = df["Budget"].astype(str).apply(budget_rank)
        else:
            df["__budget_rank"] = 3

        if df.empty:
            return None

        df = df.sort_values(by=["__near_score", "__budget_rank"], ascending=[False, True])
        top = df.iloc[0]
        hotel_name = str(top.get("Hotel", "(Unnamed)"))
        budget_label = str(top.get("Budget", ""))
        nearby = str(top.get("Nearby", ""))
        lat = top.get("Latitude", "")
        lon = top.get("Longitude", "")

        maps_link = self._maps_place_url_from_coords(hotel_name, lat, lon)
        return f"{hotel_name} ({budget_label} budget; Nearby: {nearby})\n   Maps: {maps_link}"

    def _init_geolocator(self):
        if self._geolocator is None:
            self._geolocator = Nominatim(user_agent=NOMINATIM_USER_AGENT)

    def _geolocate(self, query: str, city_hint: str = "Kolkata, India") -> Optional[Tuple[float, float]]:
        """
        Geocode using Nominatim. Returns (lat, lon) or None.
        Be polite and sleep between calls.
        """
        try:
            self._init_geolocator()
            q = f"{query}, {city_hint}"
            loc = self._geolocator.geocode(q, exactly_one=True, timeout=10)
            time.sleep(GEOCODE_SLEEP)
            if not loc:
                return None
            return (float(loc.latitude), float(loc.longitude))
        except (GeocoderTimedOut, GeocoderServiceError):
            return None
        except Exception:
            return None

    def _get_hotel_coord(self, arrival_area: str, daily_budget: int) -> Optional[Tuple[float, float]]:
        """
        Try in order:
         1) nearby_hotels_df Latitude/Longitude
         2) hotels_df Latitude/Longitude columns
         3) geocode "hotel near <arrival_area>" or arrival_area
        """
        # 1) nearby_hotels_df
        try:
            if getattr(self, "nearby_hotels_df", None) is not None and not self.nearby_hotels_df.empty:
                df = self.nearby_hotels_df.copy()
                # filter by budget when available
                if "Budget" in df.columns:
                    df = df[df["Budget"].apply(lambda x: self._budget_ok(daily_budget, str(x)))]
                # pick best match by arrival keywords
                keys = self._arrival_keywords(arrival_area)
                def score_row(r):
                    nearby_text = str(r.get("Nearby", ""))
                    t = self._norm(nearby_text)
                    return sum(1 for k in keys if k in t)
                df["__score"] = df.apply(score_row, axis=1)
                df = df.sort_values(by="__score", ascending=False)
                if not df.empty:
                    r = df.iloc[0]
                    lat = r.get("Latitude", None)
                    lon = r.get("Longitude", None)
                    if pd.notna(lat) and pd.notna(lon):
                        return (float(lat), float(lon))
        except Exception:
            pass

        # 2) hotels_df gives lat/lon columns
        try:
            lat_cols = [c for c in ["Latitude", "Lat", "latitude", "lat"] if c in self.hotels_df.columns]
            lon_cols = [c for c in ["Longitude", "Lon", "Lng", "longitude", "lon"] if c in self.hotels_df.columns]
            if lat_cols and lon_cols:
                latc = lat_cols[0]; lonc = lon_cols[0]
                # try to pick a hotel matching arrival_area (Area Focus) else first row
                if "Area Focus" in self.hotels_df.columns:
                    cand = self.hotels_df[self.hotels_df["Area Focus"].map(self._norm) == self._norm(arrival_area)]
                    r = cand.iloc[0] if not cand.empty else self.hotels_df.iloc[0]
                else:
                    r = self.hotels_df.iloc[0]
                lat = r.get(latc, None); lon = r.get(lonc, None)
                if pd.notna(lat) and pd.notna(lon):
                    return (float(lat), float(lon))
        except Exception:
            pass

        # 3) geocode fallback
        try:
            if arrival_area:
                res = self._geolocate(f"hotel near {arrival_area}")
                if res:
                    return res
            return self._geolocate("Kolkata, India")
        except Exception:
            return None

    def _init_osm_graph(self, hotel_coord: Tuple[float, float], dist_meters: int = OSM_GRAPH_DIST, network_type: str = OSM_NETWORK_TYPE) -> bool:
        """
        Build and cache an OSMnx graph centered at hotel_coord.
        Returns True if success, False otherwise.
        """
        if hotel_coord is None:
            return False
        if self._osm_graph is not None and self._hotel_coord == hotel_coord:
            return True  # already created for this hotel
        try:
            lat, lon = hotel_coord
            # Use graph_from_point; can be large — adjust dist_meters if needed
            G = ox.graph_from_point((lat, lon), dist=dist_meters, network_type=network_type)
            self._osm_graph = G
            # store nearest node for hotel
            self._osm_hotel_node = ox.distance.nearest_nodes(G, lon, lat)
            self._hotel_coord = (lat, lon)
            return True
        except Exception as e:
            # build failed
            self._osm_graph = None
            self._osm_hotel_node = None
            self._hotel_coord = None
            return False

    def _distance_from_hotel_km(self, place_name: str, place_coord: Optional[Tuple[float, float]] = None) -> Optional[float]:
        """
        Compute shortest-path distance (km) from cached hotel node to place.
        If place_coord provided, use it; otherwise try to geocode place_name.
        """
        if not self._osm_graph or self._osm_hotel_node is None:
            return None
        try:
            G = self._osm_graph
            hotel_node = self._osm_hotel_node
            # get place coordinates if not provided
            if place_coord is None:
                # first: try tourist_df / shopping_df to find coordinates in CSVs
                place_coord = None
                # tourist_df
                try:
                    lat_cols = [c for c in ["Latitude", "Lat", "latitude", "lat"] if c in self.tourist_df.columns]
                    lon_cols = [c for c in ["Longitude", "Lon", "Lng", "longitude", "lon"] if c in self.tourist_df.columns]
                    if lat_cols and lon_cols:
                        mask = self.tourist_df[self.tourist_name_col].map(self._norm) == self._norm(place_name)
                        if mask.any():
                            r = self.tourist_df[mask].iloc[0]
                            lat = r.get(lat_cols[0], None); lon = r.get(lon_cols[0], None)
                            if pd.notna(lat) and pd.notna(lon):
                                place_coord = (float(lat), float(lon))
                except Exception:
                    place_coord = None
                # shopping_df if still None
                if place_coord is None:
                    try:
                        lat_cols = [c for c in ["Latitude", "Lat", "latitude", "lat"] if c in self.shopping_df.columns]
                        lon_cols = [c for c in ["Longitude", "Lon", "Lng", "longitude", "lon"] if c in self.shopping_df.columns]
                        if lat_cols and lon_cols:
                            mask = self.shopping_df[self.shopping_name_col].map(self._norm) == self._norm(place_name)
                            if mask.any():
                                r = self.shopping_df[mask].iloc[0]
                                lat = r.get(lat_cols[0], None); lon = r.get(lon_cols[0], None)
                                if pd.notna(lat) and pd.notna(lon):
                                    place_coord = (float(lat), float(lon))
                    except Exception:
                        place_coord = None
                # final fallback: geocode by name
                if place_coord is None:
                    place_coord = self._geolocate(place_name)
            if not place_coord:
                return None
            lat, lon = place_coord
            place_node = ox.distance.nearest_nodes(G, lon, lat)
            length_m = nx.shortest_path_length(G, hotel_node, place_node, weight="length")
            return round(length_m / 1000.0, 2)
        except Exception:
            return None

    def _collect_area_meal_places(self, df: pd.DataFrame, area_focus: Optional[str],
                                  budget_category: str, meal_col_name: str) -> List[str]:
        if meal_col_name not in df.columns:
            return []

        filtered_df = df.copy()
        if self.hotels_budget_col in filtered_df.columns:
            filtered_df = filtered_df[
                filtered_df[self.hotels_budget_col].map(lambda x: self._norm(x) == self._norm(budget_category))
            ]
        if "Area Focus" in filtered_df.columns and pd.notna(area_focus):
            filtered_df = filtered_df[
                filtered_df["Area Focus"].map(self._norm) == self._norm(area_focus)
            ]

        vals = []
        for v in filtered_df[meal_col_name].dropna().astype(str):
            for p in re.split(r"/|,|\|", v):
                name = p.strip()
                if name:
                    vals.append(name)

        seen = set()
        uniq: List[str] = []
        for x in vals:
            k = self._norm(x)
            if k not in seen:
                seen.add(k)
                uniq.append(x)
        return uniq

    def _suggested_restaurants_for_meal(self, day_row: pd.Series, budget_category: str, meal: str) -> List[str]:
        meal_col = None
        for c in self.hotels_df.columns:
            if self._norm(c) == self._norm(meal):
                meal_col = c
                break
        if not meal_col:
            return []

        picks: List[str] = []

        if meal_col in day_row and pd.notna(day_row[meal_col]):
            for p in re.split(r"/|,|\|", str(day_row[meal_col])):
                p = p.strip()
                if p and p not in picks:
                    picks.append(p)

        area_val = day_row.get("Area Focus", None)
        more = self._collect_area_meal_places(self.hotels_df, area_val, budget_category, meal_col)
        for m in more:
            if m not in picks:
                picks.append(m)

        if len(picks) < 3:
            for v in self.hotels_df[meal_col].dropna().astype(str):
                for p in re.split(r"/|,|\|", v):
                    p = p.strip()
                    if p and p not in picks:
                        picks.append(p)
                        if len(picks) >= 4:
                            break
                if len(picks) >= 4:
                    break

        return picks[:4]

    def get_single_hotel_recommendation(self, budget: str, arrivals: str) -> str:
        if "Sealdah Railway Station" in arrivals and "Esplanade Bus Stand" in arrivals:
            arrival_key= "Sealdah Railway Station/ Esplanade Bus Stand"
        else:
            arrival_key=arrivals
        df = self.hotelsRec_df[
            (self.hotelsRec_df["Nearby"].str.strip().str.lower() == arrival_key.strip().lower()) &
            (self.hotelsRec_df["Budget"].str.strip().str.lower() == budget.strip().lower())
        ]
        if df.empty:
            return f"No hotel found for arrival='{arrivals}' and budget='{budget}'."
        row = df.iloc[0]
        hotel_name = row["Hotel"]
        lat, lon = row["Latitude"], row["Longitude"]
        maps_link = self._maps_place_url_from_coords(hotel_name, lat, lon)
        return f"{hotel_name} ({budget} budget; Nearby: {arrivals})\n   Maps: {maps_link}"

    def generate_itinerary(
            self,
            days: int,
            budget_level:str,
            user_budget_value: int,
            group_type: str,
            start_location: str,
            transportation_mode: str,
            arrival,
            arrival_area: Optional[str] = None
        ) -> Dict[str, Any]:
            
            user_budget_category = self._map_budget_to_category(user_budget_value)
            arrival_area_guess = arrival_area or start_location or "Kolkata, India"
            hotel_coord = self._get_hotel_coord(arrival_area_guess, user_budget_value)
            
            osm_ready = False
            if hotel_coord:
                osm_ready = self._init_osm_graph(hotel_coord, dist_meters=OSM_GRAPH_DIST, network_type=OSM_NETWORK_TYPE)

            day_rows = self.hotels_df.copy()

            if "Group Type" in day_rows.columns:
                day_rows = day_rows[day_rows["Group Type"].map(self._norm).str.contains(self._norm(group_type), na=False)]
            if self.hotels_budget_col in day_rows.columns:
                day_rows = day_rows[day_rows[self.hotels_budget_col].map(self._norm) == self._norm(user_budget_category)]

            if day_rows.empty:
                day_rows = self.hotels_df.copy()

            day_rows = day_rows.head(days).reset_index(drop=True)

            if day_rows.empty:
                return {
                    "meta": {
                        "error": "No rows available to build itinerary. Check input CSVs."
                    },
                    "days": []
                }

            itinerary_obj: Dict[str, Any] = {
                "meta": {
                    "days": days,
                    "budget_value_inr": user_budget_value,
                    "budget_category": user_budget_category,
                    "group_type": group_type,
                    "start_location": start_location,
                    "transportation_mode": transportation_mode,
                    "arrival_area": arrival_area_guess,
                    "osm_ready": osm_ready,
                },
                "days": []
            }

            for i, row in day_rows.iterrows():
                day_num = i + 1
                area = row.get("Area Focus", None)

                spots = self._parse_spots(row.get("Spots (Order & Duration)", ""))
                location_order = [start_location]

                day_items: List[Dict[str, Any]] = []
                for s in spots:
                    nm = s["name"]
                    kind = s["kind"]
                    mins = s["minutes"]

                    location_order.append(nm)
                    place_coord = None
                    try:
                        lat_cols = [c for c in ["Latitude", "Lat", "latitude", "lat"] if c in self.tourist_df.columns]
                        lon_cols = [c for c in ["Longitude", "Lon", "Lng", "longitude", "lon"] if c in self.tourist_df.columns]
                        if lat_cols and lon_cols:
                            mask = self.tourist_df[self.tourist_name_col].map(self._norm) == self._norm(nm)
                            if mask.any():
                                r = self.tourist_df[mask].iloc[0]
                                lat = r.get(lat_cols[0], None); lon = r.get(lon_cols[0], None)
                                if pd.notna(lat) and pd.notna(lon):
                                    place_coord = (float(lat), float(lon))
                    except Exception:
                        place_coord = None

                    dkm = None
                    if osm_ready:
                        dkm = self._distance_from_hotel_km(nm, place_coord)

                    if kind == "shopping":
                        desc, acts = self._safe_info(self.shopping_df, nm, "shopping data")
                    elif kind in ("lunch", "dinner"):
                        desc, acts = "", ""
                    else:
                        desc, acts = self._safe_info(self.tourist_df, nm, "tourist spot data")

                    day_items.append({
                        "type": kind,  # visit / shopping / lunch / dinner
                        "name": nm,
                        "duration_minutes": mins,
                        "distance_km_from_hotel": dkm,
                        "description": desc,
                        "top_activities": acts
                    })  # --------- Meal Plan ----------
                alloc = self._parse_meal_alloc(row.get("Meals Time Allocation", ""))
                meals_block: Dict[str, Any] = {}
                for meal_label in ["Breakfast", "Lunch", "Snacks", "Dinner"]:
                    minutes = alloc.get(meal_label, 0)
                    suggestions = self._suggested_restaurants_for_meal(row, user_budget_category, meal_label)
                    sugg_list: List[Dict[str, Any]] = []
                    for rname in suggestions:
                        rcoord = None
                        try:
                            lat_cols = [c for c in ["Latitude", "Lat", "latitude", "lat"] if c in self.hotels_df.columns]
                            lon_cols = [c for c in ["Longitude", "Lon", "Lng", "longitude", "lon"] if c in self.hotels_df.columns]
                            if lat_cols and lon_cols:
                                mask = self.hotels_df[self.hotels_name_col].map(self._norm) == self._norm(rname)
                                if mask.any():
                                    r = self.hotels_df[mask].iloc[0]
                                    lat = r.get(lat_cols[0], None); lon = r.get(lon_cols[0], None)
                                    if pd.notna(lat) and pd.notna(lon):
                                        rcoord = (float(lat), float(lon))
                        except Exception:
                            rcoord = None

                        if rcoord is None:
                            try:
                                lat_cols = [c for c in ["Latitude", "Lat", "latitude", "lat"] if c in self.tourist_df.columns]
                                lon_cols = [c for c in ["Longitude", "Lon", "Lng", "longitude", "lon"] if c in self.tourist_df.columns]
                                if lat_cols and lon_cols:
                                    mask = self.tourist_df[self.tourist_name_col].map(self._norm) == self._norm(rname)
                                    if mask.any():
                                        r = self.tourist_df[mask].iloc[0]
                                        lat = r.get(lat_cols[0], None); lon = r.get(lon_cols[0], None)
                                        if pd.notna(lat) and pd.notna(lon):
                                            rcoord = (float(lat), float(lon))
                            except Exception:
                                rcoord = None

                        dkm = self._distance_from_hotel_km(rname, rcoord) if osm_ready else None
                        sugg_list.append({
                            "name": rname,
                            "distance_km_from_hotel": dkm
                        })

                    meals_block[meal_label] = {
                        "allocated_minutes": minutes,
                        "suggested_restaurants": sugg_list
                    }

                maps_url = self._generate_maps_url(location_order)
                trans_tips = self._get_transportation_tips(user_budget_value, group_type, transportation_mode)

                itinerary_obj["days"].append({
                    "day_number": day_num,
                    "area_focus": area,
                    "items": day_items,
                    "meals": meals_block,
                    "maps_route_url": maps_url,
                    "transportation_tips": trans_tips
                })

            return itinerary_obj

def ask_question(prompt: str, valid_options: Optional[List[str]] = None) -> str:
    while True:
        try:
            user_input = input(prompt).strip()
            if not user_input:
                continue
            if valid_options and user_input.title() not in valid_options:
                print(f"Please enter one of the following: {', '.join(valid_options)}")
                continue
            return user_input.title() if valid_options else user_input
        except Exception as e:
            print(f"Invalid input. Please try again. ({e})")

def main():
    generator = KolkataItineraryGenerator(
        hotels_csv="kolkata_restaurants_hotels.csv",
        tourist_csv="kolkata_tourist_spots.csv",
        shopping_csv="shopping.csv",
        nearby_hotels_csv="hotels_kolkata.csv"   
    )

    try:
        days = int(ask_question("Enter duration of your trip: "))
    except (ValueError, TypeError):
        print("Invalid number of days. Please enter a number.")
        return

    try:
        user_budget_value = int(ask_question("Enter your estimated daily budget (in INR): "))
    except (ValueError, TypeError):
        print("Invalid budget. Please enter a number.")
        return

    group_type = ask_question("Enter group type (Solo / Family / Friends / Couple): ", ["Solo", "Family", "Friends", "Couple"])
    start_location = ask_question("Enter your starting location (e.g., Hotel, Relatives' House): ")

    travel_mode = ask_question("How are you reaching Kolkata? (Flight / Train / Bus): ", ["Flight", "Train", "Bus"])

    if travel_mode == "Flight":
        arrival_area = "Dum Dum"
    elif travel_mode == "Train":
        station = ask_question("Which station? (Howrah / Sealdah): ", ["Howrah", "Sealdah"])
        arrival_area = station
    else:
        arrival_area = "Esplanade"

    transportation_mode = ask_question("Enter preferred transportation mode (Public Transport / Ride-Hailing / Private Cab): ", ["Public Transport", "Ride-Hailing", "Private Cab"])

    hotel_choice = generator.get_single_hotel_recommendation(user_budget_value, group_type, arrival_area)

    print(f"\n🏨 Recommended Hotel for your stay: {hotel_choice}\n")

    # pass arrival_area so geocoding/hotel selection gets the right hint
    generator.generate_itinerary(days, user_budget_value, group_type, start_location, transportation_mode, arrival_area=arrival_area)

def fetch_itinerary(days, user_budget_value, group_type, arrival, start_location="Hotel") :
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    generator = KolkataItineraryGenerator(
        hotels_csv=os.path.join(BASE_DIR, "kolkata_restaurants_hotels.csv"),
        hotel_recommendation_csv=os.path.join(BASE_DIR, "hotels_kolkata.csv"),
        tourist_csv=os.path.join(BASE_DIR, "kolkata_tourist_spots.csv"),
        shopping_csv=os.path.join(BASE_DIR, "shopping.csv"),
        nearby_hotels_csv=os.path.join(BASE_DIR, "hotels_kolkata.csv")   
    )

    arrival_mapping = {
        "Dumdum Airport": "Dum Dum",
        "Howrah Railway Station": "Howrah",
        "Sealdah Railway Station": "Sealdah",
        "Esplanade Bus Stand": "Esplanade"
    }
    arrival_area = arrival_mapping.get(arrival, "Esplanade")

    budget_mapping = {
        "High": "Private Cab",
        "Mid": "Ride-Hailing",
        "Low": "Public Transport"
    }

    if user_budget_value >= 5000:
        budget_level = "High"
    elif user_budget_value >= 1500:
        budget_level = "Mid"
    else:
        budget_level = "Low"

    transportation_mode = budget_mapping[budget_level]
    hotel_choice = generator.get_single_hotel_recommendation(budget_level, arrival)

    itinerary=generator.generate_itinerary(days, budget_level, user_budget_value, group_type, start_location, transportation_mode, arrival, arrival_area)
    return {
        "Hotel": hotel_choice,
        "Itinerary": itinerary
    }

if __name__ == "__main__":
    main()
