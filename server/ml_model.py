"""
ML inference module — loads LightGBM and exposes predict_ml().
Feature extraction is identical to ML/predict_url.py.
Returns None if ANY feature fails to extract (triggers DL-only fallback).
"""
from __future__ import annotations

import math
import re
import socket
import ssl
import urllib.parse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import joblib
import pandas as pd

try:
    import requests
    requests.packages.urllib3.disable_warnings()
    _REQUESTS_OK = True
except ImportError:
    _REQUESTS_OK = False

try:
    from bs4 import BeautifulSoup
    _BS4_OK = True
except ImportError:
    _BS4_OK = False

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT       = Path(__file__).parent.parent
_MODEL_PATH = _ROOT / "ML" / "lightgbm_model.pkl"

# ── Feature order (must match training) ───────────────────────────────────────
FEATURE_ORDER = [
    "url_has_ip", "url_length", "url_length_flag", "uses_url_shortener",
    "url_has_at_symbol", "url_double_slash_redirect", "url_prefix_suffix_dash",
    "url_subdomain_count", "url_subdomain_flag", "ssl_final_state",
    "domain_reg_length_flag", "favicon_external", "non_standard_port",
    "https_in_domain_token", "url_special_char_count", "url_has_suspicious_kw",
    "url_dot_count", "dns_record_exists", "domain_age_flag", "domain_length",
    "domain_has_numerics", "tld_suspicious", "web_traffic_flag", "pagerank_flag",
    "google_index_flag", "links_pointing_flag", "statistical_report_flag",
    "domain_entropy", "domain_dot_count", "html_external_req_ratio",
    "html_anchor_url_ratio", "html_links_meta_script", "html_sfh_suspicious",
    "html_submits_to_email", "html_abnormal_url", "html_redirect_count",
    "html_status_bar_custom", "html_disables_right_click", "html_uses_popup",
    "html_uses_iframe", "html_form_count", "sec_domain_age_flag",
    "sec_dns_record_exists", "sec_web_traffic_flag", "sec_links_pointing_flag",
    "sec_statistical_report_flag", "sec_ssl_state", "sec_tld_suspicious",
    "sec_has_suspicious_kw",
]

URL_SHORTENERS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "buff.ly",
    "adf.ly", "short.link", "rebrand.ly", "cutt.ly", "tiny.cc",
    "is.gd", "cli.gs", "su.pr", "twurl.nl", "snurl.com", "short.to",
}

SUSPICIOUS_KEYWORDS = {
    "login", "secure", "account", "update", "banking", "confirm",
    "verify", "paypal", "ebay", "amazon", "signin", "password",
    "credential", "wallet", "billing", "support", "alert", "suspend",
    "unlock", "access", "validation", "authenticate",
}

SUSPICIOUS_TLDS = {
    ".tk", ".ml", ".ga", ".cf", ".gq", ".xyz", ".top", ".club",
    ".online", ".site", ".info", ".live", ".icu", ".buzz", ".fun",
    ".work", ".click", ".link", ".win", ".bid", ".loan",
}

STANDARD_PORTS = {"", "80", "443"}
IP_PATTERN     = re.compile(r"((\d{1,3}\.){3}\d{1,3})|(\[[\da-fA-F:]+\])")


# ── Module-level state ────────────────────────────────────────────────────────
_model = None


def load_ml_model() -> None:
    global _model
    _model = joblib.load(_MODEL_PATH)
    print(f"[ML] LightGBM loaded from {_MODEL_PATH}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    freq: dict[str, int] = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    n = len(s)
    return -sum((v / n) * math.log2(v / n) for v in freq.values())


def _parse_url(url: str):
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    return urllib.parse.urlparse(url), url


def _check_dns(hostname: str) -> int:
    try:
        socket.setdefaulttimeout(5)
        socket.gethostbyname(hostname)
        return 1
    except Exception:
        return -1


def _check_ssl(hostname: str) -> int:
    ctx = ssl.create_default_context()
    try:
        with ctx.wrap_socket(
            socket.create_connection((hostname, 443), timeout=5),
            server_hostname=hostname,
        ):
            return 1
    except ssl.SSLCertVerificationError:
        return -1
    except Exception:
        return 0


def _fetch_html(url: str) -> tuple[str | None, int]:
    if not _REQUESTS_OK:
        return None, 0
    try:
        resp = requests.get(
            url, timeout=10, verify=False, allow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        return resp.text, len(resp.history)
    except Exception:
        return None, 0


# ── Feature extraction ────────────────────────────────────────────────────────

def _extract_features(url: str) -> dict:
    """
    Extract all 49 features.
    Raises an exception if any core (non-API-dependent) feature fails.
    API-dependent features always return -1 by design.
    """
    parsed, url = _parse_url(url)

    scheme   = parsed.scheme.lower()
    hostname = parsed.hostname or ""
    port     = str(parsed.port) if parsed.port else ""
    path     = parsed.path
    query    = parsed.query

    parts      = hostname.split(".")
    tld        = ("." + parts[-1]) if len(parts) >= 2 else ""
    sld        = parts[-2]         if len(parts) >= 2 else hostname
    reg_domain = sld + tld
    subdomains = parts[:-2]        if len(parts) > 2 else []

    full_url_lower = url.lower()

    # ── URL-level features ────────────────────────────────────────────────────
    url_has_ip                = 1 if IP_PATTERN.search(hostname) else 0
    url_length                = len(url)
    url_length_flag           = 1 if url_length > 75 else 0
    uses_url_shortener        = 1 if hostname in URL_SHORTENERS else 0
    url_has_at_symbol         = 1 if "@" in url else 0

    path_query                = path + ("?" + query if query else "")
    url_double_slash_redirect = 1 if "//" in path_query else 0

    url_prefix_suffix_dash    = 1 if "-" in hostname else 0
    url_subdomain_count       = len(subdomains)
    url_subdomain_flag        = 1 if url_subdomain_count >= 2 else 0

    special_chars             = re.findall(
        r"[!@#$%^&*()+=\[\]{};'\":<>?,\\|`~]", path + query
    )
    url_special_char_count    = len(special_chars)
    url_has_suspicious_kw     = 1 if any(k in full_url_lower for k in SUSPICIOUS_KEYWORDS) else 0
    url_dot_count             = url.count(".")

    # ── Domain-level features ─────────────────────────────────────────────────
    domain_length           = len(reg_domain)
    domain_has_numerics     = 1 if any(c.isdigit() for c in sld) else 0
    domain_entropy          = round(_shannon_entropy(reg_domain), 4)
    domain_dot_count        = hostname.count(".")
    https_in_domain_token   = 1 if "https" in hostname.lower() else 0
    tld_suspicious          = 1 if tld.lower() in SUSPICIOUS_TLDS else 0
    non_standard_port       = 1 if port in STANDARD_PORTS else 0

    ssl_final_state         = _check_ssl(hostname) if scheme == "https" else 0
    dns_record_exists       = _check_dns(hostname)

    # API-dependent — always -1 (model trained with this)
    domain_reg_length_flag  = -1
    domain_age_flag         = -1
    web_traffic_flag        = -1
    pagerank_flag           = -1
    google_index_flag       = -1
    links_pointing_flag     = -1
    statistical_report_flag = -1

    # ── HTML features (default -1 until fetched) ──────────────────────────────
    html_external_req_ratio   = -1
    html_anchor_url_ratio     = -1
    html_links_meta_script    = -1
    html_sfh_suspicious       = -1
    html_submits_to_email     = -1
    html_abnormal_url         = -1
    html_redirect_count       = -1
    html_status_bar_custom    = -1
    html_disables_right_click = -1
    html_uses_popup           = -1
    html_uses_iframe          = -1
    html_form_count           = -1
    favicon_external          = -1

    if _REQUESTS_OK and dns_record_exists == 1:
        html_content, html_redirect_count = _fetch_html(url)

        if html_content and _BS4_OK:
            soup    = BeautifulSoup(html_content, "html.parser")
            html_lc = html_content.lower()

            fav = soup.find("link", rel=lambda r: r and "icon" in " ".join(r).lower())
            if fav and fav.get("href"):
                fav_domain   = urllib.parse.urlparse(fav["href"]).hostname or ""
                favicon_external = 0 if (not fav_domain or hostname in fav_domain) else 1
            else:
                favicon_external = 0

            all_res = (soup.find_all("img", src=True) +
                       soup.find_all("script", src=True) +
                       soup.find_all("link", href=True))
            ext_res = [r for r in all_res
                       if urllib.parse.urlparse(
                           r.get("src") or r.get("href", "")
                       ).hostname not in ("", None, hostname)]
            html_external_req_ratio = round(len(ext_res) / len(all_res), 4) if all_res else 0

            anchors     = soup.find_all("a", href=True)
            ext_anchors = [a for a in anchors
                           if urllib.parse.urlparse(a["href"]).hostname
                           not in ("", None, hostname)]
            html_anchor_url_ratio = round(len(ext_anchors) / len(anchors), 4) if anchors else 0

            total_links  = len(soup.find_all("a"))
            total_script = len(soup.find_all("script"))
            html_links_meta_script = 1 if (total_links + total_script) == 0 else 0

            forms       = soup.find_all("form")
            html_form_count = len(forms)
            sfh_susp    = 0
            sub_email   = 0
            for form in forms:
                action = (form.get("action") or "").strip().lower()
                if action in ("", "about:blank") or (
                    urllib.parse.urlparse(action).hostname not in ("", None, hostname)
                ):
                    sfh_susp = 1
                if action.startswith("mailto:"):
                    sub_email = 1
            html_sfh_suspicious   = sfh_susp
            html_submits_to_email = sub_email

            html_abnormal_url         = 0 if hostname in html_lc else 1
            html_status_bar_custom    = 1 if "onmouseover"  in html_lc else 0
            html_disables_right_click = 1 if "contextmenu"  in html_lc else 0
            html_uses_popup           = 1 if "window.open"  in html_lc else 0
            html_uses_iframe          = 1 if soup.find("iframe") else 0

        elif html_content:
            html_lc = html_content.lower()
            html_status_bar_custom    = 1 if "onmouseover"    in html_lc else 0
            html_disables_right_click = 1 if "event.button==2" in html_lc else 0
            html_uses_popup           = 1 if "window.open"    in html_lc else 0
            html_uses_iframe          = 1 if "<iframe"         in html_lc else 0
            html_form_count           = html_lc.count("<form")

    # ── Secondary mirrors ─────────────────────────────────────────────────────
    sec_domain_age_flag         = domain_age_flag
    sec_dns_record_exists       = dns_record_exists
    sec_web_traffic_flag        = -1
    sec_links_pointing_flag     = -1
    sec_statistical_report_flag = -1
    sec_ssl_state               = ssl_final_state
    sec_tld_suspicious          = tld_suspicious
    sec_has_suspicious_kw       = url_has_suspicious_kw

    return {
        "url_has_ip":                  url_has_ip,
        "url_length":                  url_length,
        "url_length_flag":             url_length_flag,
        "uses_url_shortener":          uses_url_shortener,
        "url_has_at_symbol":           url_has_at_symbol,
        "url_double_slash_redirect":   url_double_slash_redirect,
        "url_prefix_suffix_dash":      url_prefix_suffix_dash,
        "url_subdomain_count":         url_subdomain_count,
        "url_subdomain_flag":          url_subdomain_flag,
        "ssl_final_state":             ssl_final_state,
        "domain_reg_length_flag":      domain_reg_length_flag,
        "favicon_external":            favicon_external,
        "non_standard_port":           non_standard_port,
        "https_in_domain_token":       https_in_domain_token,
        "url_special_char_count":      url_special_char_count,
        "url_has_suspicious_kw":       url_has_suspicious_kw,
        "url_dot_count":               url_dot_count,
        "dns_record_exists":           dns_record_exists,
        "domain_age_flag":             domain_age_flag,
        "domain_length":               domain_length,
        "domain_has_numerics":         domain_has_numerics,
        "tld_suspicious":              tld_suspicious,
        "web_traffic_flag":            web_traffic_flag,
        "pagerank_flag":               pagerank_flag,
        "google_index_flag":           google_index_flag,
        "links_pointing_flag":         links_pointing_flag,
        "statistical_report_flag":     statistical_report_flag,
        "domain_entropy":              domain_entropy,
        "domain_dot_count":            domain_dot_count,
        "html_external_req_ratio":     html_external_req_ratio,
        "html_anchor_url_ratio":       html_anchor_url_ratio,
        "html_links_meta_script":      html_links_meta_script,
        "html_sfh_suspicious":         html_sfh_suspicious,
        "html_submits_to_email":       html_submits_to_email,
        "html_abnormal_url":           html_abnormal_url,
        "html_redirect_count":         html_redirect_count,
        "html_status_bar_custom":      html_status_bar_custom,
        "html_disables_right_click":   html_disables_right_click,
        "html_uses_popup":             html_uses_popup,
        "html_uses_iframe":            html_uses_iframe,
        "html_form_count":             html_form_count,
        "sec_domain_age_flag":         sec_domain_age_flag,
        "sec_dns_record_exists":       sec_dns_record_exists,
        "sec_web_traffic_flag":        sec_web_traffic_flag,
        "sec_links_pointing_flag":     sec_links_pointing_flag,
        "sec_statistical_report_flag": sec_statistical_report_flag,
        "sec_ssl_state":               sec_ssl_state,
        "sec_tld_suspicious":          sec_tld_suspicious,
        "sec_has_suspicious_kw":       sec_has_suspicious_kw,
    }


# ── Public API ────────────────────────────────────────────────────────────────

def predict_ml(url: str) -> float | None:
    """
    Return phishing probability in [0, 1], or None if any feature fails.
    None triggers DL-only fallback in the fusion layer.
    """
    try:
        features = _extract_features(url)

        # Verify every expected feature is present — missing key = extraction failure
        for feat in FEATURE_ORDER:
            if feat not in features:
                print(f"[ML] Feature missing: {feat} — falling back to DL only")
                return None

        df   = pd.DataFrame([features])[FEATURE_ORDER]
        prob = float(_model.predict_proba(df)[0][1])
        return prob

    except Exception as exc:
        print(f"[ML] Feature extraction failed ({exc}) — falling back to DL only")
        return None
