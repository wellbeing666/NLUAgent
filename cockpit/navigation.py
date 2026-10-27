"""高德Web服务适配器：Key仅在服务端，坐标统一为GCJ-02（经度在前）。"""

import base64
import json
import math
import os
import re
import unicodedata
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener


STRATEGIES = {"recommended": (32, 0), "avoid_congestion": (33, 1),
              "less_fee": (36, 2), "no_highway": (35, 3)}


def distance_m(origin, destination):
    """坐标间近似球面距离；不是驾车距离，也不用于声称最短驾车时间。"""
    lng1, lat1 = map(math.radians, map(float, coordinate(origin).split(",")))
    lng2, lat2 = map(math.radians, map(float, coordinate(destination).split(",")))
    a = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lng2 - lng1) / 2) ** 2
    return 6371000 * 2 * math.asin(math.sqrt(min(1, max(0, a))))


def matching_places(places, keywords, origin):
    """有限名称匹配，排除并未请求的附属设施，再按距离排序。"""
    def normalized(value):
        return re.sub(r"\s+", "", unicodedata.normalize("NFKC", value)).casefold()

    query = normalized(keywords)
    candidates, seen = [], set()
    for place in places:
        name = normalized(place["name"])
        # 例如：导航到北京南站，不应因停车场更近而选中北京南站停车场。
        # “速8酒店(北京南站店)”只是地址提及，不能当作“北京南站”。
        if not name.startswith(query):
            continue
        # 明确交通枢纽名称只接收同名本体；入口/门店等要求用户在关键词中明确。
        if query.endswith(("站", "机场", "航站楼")) and name != query:
            continue
        facilities = ("停车场", "售票处", "售票厅", "候车厅", "公交站", "地铁站", "卫生间", "便利店", "充电站",
                      "酒店", "旅馆", "宾馆", "公寓", "民宿")
        if any(word in name and word not in query for word in facilities):
            continue
        identity = (place["name"], place["location"])
        if identity in seen:
            continue
        seen.add(identity)
        candidates.append({**place, "distance_m": round(distance_m(origin, place["location"]), 1)})
    return sorted(candidates, key=lambda p: (p["distance_m"], p["name"], p.get("id", ""), p["location"]))


class NavigationError(Exception):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


def amap_key():
    # 离线核心仍可只用标准库；已安装dotenv时沿用项目配置方式。
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    except ImportError:
        pass
    return os.getenv("AMAP_API_KEY", "").strip()


def text_value(value, label, limit=100, optional=False):
    if not isinstance(value, str) or len(value.strip()) > limit or (not optional and not value.strip()):
        raise ValueError(f"{label}必须是{'不超过' if optional else '1～'}{limit}个字符的文本。")
    return value.strip()


def coordinate(value):
    if not isinstance(value, str) or len(value) > 60 or len(value.split(",")) != 2:
        raise ValueError("坐标必须为GCJ-02经度,纬度。")
    try:
        lng, lat = map(float, value.split(","))
    except ValueError:
        raise ValueError("坐标必须为GCJ-02经度,纬度。") from None
    if not math.isfinite(lng) or not math.isfinite(lat) or not -180 <= lng <= 180 or not -90 <= lat <= 90:
        raise ValueError("坐标超出经纬度范围。")
    return f"{lng:.6f},{lat:.6f}"


def point(value):
    if not isinstance(value, dict) or set(value) - {"name", "location", "id"}:
        raise ValueError("地点需包含name和location，可选id。")
    result = {"name": text_value(value.get("name"), "地点名称"),
              "location": coordinate(value.get("location"))}
    if value.get("id"):
        result["id"] = text_value(value["id"], "地点ID", 80)
    return result


def navigation_url(origin, destination, strategy="recommended", callnative=False):
    # URI策略与REST策略编号不同，不能直接复用32/33等值。
    if strategy not in STRATEGIES:
        raise ValueError("不支持的路线偏好。")
    origin, destination = point(origin), point(destination)
    return "https://uri.amap.com/navigation?" + urlencode({
        "from": origin["location"] + "," + origin["name"].replace(",", " "),
        "to": destination["location"] + "," + destination["name"].replace(",", " "),
        "mode": "car", "policy": STRATEGIES[strategy][1], "src": "CabinLab",
        "callnative": int(callnative),
    })


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # 不把带Key的请求重定向到其他地址。


class AmapClient:
    def __init__(self, key=None, transport=None):
        self._key = amap_key() if key is None else key
        self._transport = transport or self._http

    @staticmethod
    def _http(url):
        try:
            request = Request(url, headers={"User-Agent": "CabinLab/1.0"})
            with build_opener(NoRedirect()).open(request, timeout=8) as response:
                raw = response.read(4 * 1024 * 1024 + 1)
                content_type = response.headers.get_content_type()
            if len(raw) > 4 * 1024 * 1024:
                raise NavigationError("AMAP_RESPONSE", "高德返回内容过大，请缩短路线后重试。")
            return raw, content_type
        except (TimeoutError, URLError, HTTPError, OSError):
            # urllib异常中可能含完整URL和Key，禁止直接回传或记录。
            raise NavigationError("AMAP_NETWORK", "高德请求超时或连接失败，请检查网络后重试。") from None

    def _request(self, endpoint, params, image=False):
        if not self._key:
            raise NavigationError("AMAP_NOT_CONFIGURED", "请在本地.env配置AMAP_API_KEY（高德Web服务类型Key），然后重启服务。")
        url = "https://restapi.amap.com" + endpoint + "?" + urlencode({**params, "key": self._key})
        raw, content_type = self._transport(url)
        if image and content_type in {"image/png", "image/jpeg"}:
            return {"image": f"data:{content_type};base64," + base64.b64encode(raw).decode("ascii")}
        try:
            data = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            raise NavigationError("AMAP_RESPONSE", "高德返回了无法解析的响应，请稍后重试。") from None
        if not isinstance(data, dict):
            raise NavigationError("AMAP_RESPONSE", "高德响应格式异常。")
        if str(data.get("status")) != "1":
            code = str(data.get("infocode", ""))
            messages = {"10001": "高德Key无效，请检查Key类型和配置。",
                        "10003": "高德调用次数已达限制，请检查配额。",
                        "10004": "高德请求过于频繁，请稍后再试。",
                        "10005": "高德IP白名单或访问权限不匹配。",
                        "10007": "高德签名或Key权限校验失败。",
                        "10009": "该Key没有此高德接口的使用权限，请检查服务授权。",
                        "10044": "高德账号或服务权限受限，请检查控制台。"}
            raise NavigationError("AMAP_SERVICE", messages.get(code, "高德服务拒绝请求，请检查服务权限与配额。"))
        if image:
            raise NavigationError("AMAP_RESPONSE", "高德未返回地图图片，路线仍可通过路书或高德链接查看。")
        return data

    @staticmethod
    def _places(data, limit=25):
        pois = data.get("pois")
        if not isinstance(pois, list):
            raise NavigationError("AMAP_RESPONSE", "高德地点结果格式异常。")
        places = []
        for poi in pois[:limit]:
            try:
                p = point({"name": poi.get("name"), "location": poi.get("location"), "id": poi.get("id")})
            except (ValueError, AttributeError):
                continue
            p["address"] = " ".join(poi.get(k) for k in ["pname", "cityname", "adname", "address"]
                                    if isinstance(poi.get(k), str))[:300]
            places.append(p)
        return places

    def search(self, keywords, city):
        keywords = text_value(keywords, "搜索关键词")
        city = text_value(city, "搜索城市", 40)
        data = self._request("/v3/place/text", {"keywords": keywords, "city": city,
                             "citylimit": "true", "offset": 8, "page": 1, "extensions": "base"})
        return {"provider": "amap", "coordinate_system": "GCJ-02", "places": self._places(data, 8),
                "note": "请选择具体地点，未自动选中首个搜索结果。"}

    def nearest(self, keywords, origin):
        keywords = text_value(keywords, "目的地名称")
        keywords = re.sub(r"^(?:(?:附近|最近)(?:的)?)+", "", keywords).strip()
        if not keywords:
            raise ValueError("请给出具体的目的地名称。")
        origin = coordinate(origin)
        data = self._request("/v3/place/around", {"keywords": keywords, "location": origin,
                             "radius": 50000, "sortrule": "distance", "offset": 25,
                             "page": 1, "extensions": "base"})
        candidates = matching_places(self._places(data), keywords, origin)
        scope = "within_50km"
        if not candidates:
            # 明确的远距离目的地不局限于本地城市；全国搜索也只处理本次返回候选。
            data = self._request("/v3/place/text", {"keywords": keywords, "offset": 25,
                                 "page": 1, "extensions": "base"})
            candidates = matching_places(self._places(data), keywords, origin)
            scope = "national_candidates"
        if not candidates:
            raise NavigationError("AMAP_NO_MATCH", "没有找到名称匹配的目的地，请补充城市、门店或具体地点名称。")
        return {"query": keywords, "selected": candidates[0], "candidates": candidates[:5],
                "candidate_count": len(candidates), "search_scope": scope,
                "distance_metric": "approx_straight_line",
                "basis": "在本次返回且名称匹配的候选中，选择坐标近似直线距离最近者；不是最短驾车距离或全量地点检索。"}

    def driving(self, origin, destination, strategy="recommended"):
        origin, destination = point(origin), point(destination)
        if strategy not in STRATEGIES:
            raise ValueError("不支持的路线偏好。")
        if origin["location"] == destination["location"]:
            raise ValueError("起点和终点不能相同。")
        params = {"origin": origin["location"], "destination": destination["location"],
                  "strategy": STRATEGIES[strategy][0], "show_fields": "cost,polyline,navi"}
        if origin.get("id"):
            params["origin_id"] = origin["id"]
        if destination.get("id"):
            params["destination_id"] = destination["id"]
        data = self._request("/v5/direction/driving", params)
        try:
            paths = data["route"]["paths"]
            if not isinstance(paths, list) or not paths:
                raise KeyError()
            routes = []
            for path in paths[:3]:
                cost = path.get("cost", {})
                distance, seconds = float(path["distance"]), float(cost["duration"])
                tolls = float(cost.get("tolls") or 0)
                if not all(math.isfinite(n) and n >= 0 for n in [distance, seconds, tolls]):
                    raise ValueError()
                steps, geometry = [], []
                raw_steps = path.get("steps", [])
                if not isinstance(raw_steps, list) or len(raw_steps) > 1000:
                    raise ValueError()
                for step in raw_steps:
                    step_distance = float(step.get("step_distance") or 0)
                    if not math.isfinite(step_distance) or step_distance < 0:
                        raise ValueError()
                    steps.append({"instruction": str(step.get("instruction", ""))[:300],
                                  "road": str(step.get("road_name") or "")[:100],
                                  "distance_m": step_distance})
                    polyline = step.get("polyline")
                    if isinstance(polyline, str) and polyline:
                        geometry.extend(coordinate(p) for p in polyline.split(";"))
                routes.append({"name": str(path.get("strategy") or "高德驾车路线")[:100],
                               "distance_km": round(distance / 1000, 2),
                               "duration_minutes": round(seconds / 60, 1), "tolls_yuan": tolls,
                               "steps": steps, "geometry": geometry})
        except (KeyError, ValueError, TypeError, AttributeError):
            raise NavigationError("AMAP_NO_ROUTE", "高德未返回完整可用的驾车路线，请更换具体起终点后重试。") from None
        return {"provider": "amap", "coordinate_system": "GCJ-02", "origin": origin,
                "destination": destination, "strategy": strategy, "routes": routes,
                "web_url": navigation_url(origin, destination, strategy),
                "app_url": navigation_url(origin, destination, strategy, callnative=True),
                "note": "真实高德路线快照；导航链接会由高德重新算路，不保证与所选备选路线完全一致。未接入真实车辆定位或耗电模型。"}

    def static_map(self, plan, route):
        # 图片概览采用稀疏采样控制URL长度，完整转向信息仍来自原路线。
        geometry = route["geometry"]
        sampled = [geometry[round(i * (len(geometry) - 1) / 79)] for i in range(80)] if len(geometry) > 80 else geometry
        params = {"size": "800*450", "scale": 1,
                  "markers": f"mid,0x43b894,A:{plan['origin']['location']}|mid,0x7159e6,B:{plan['destination']['location']}"}
        if len(sampled) >= 2:
            params["paths"] = "5,0x7159e6,1,,:" + ";".join(sampled)
        return self._request("/v3/staticmap", params, image=True)
