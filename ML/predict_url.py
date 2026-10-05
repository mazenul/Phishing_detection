import re
import math
import socket
import ssl
import warnings
import urllib.parse
warnings.filterwarnings("ignore")

import pandas as pd
import joblib

try:
    import requests
    requests.packages.urllib3.disable_warnings()
    REQUESTS_OK = True
except ImportError:
    REQUESTS_OK = False

try:
    from bs4 import BeautifulSoup
    BS4_OK = True
except ImportError:
    BS4_OK = False

# ── Constants ─────────────────────────────────────────────────────────────────

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

IP_PATTERN = re.compile(r"((\d{1,3}\.){3}\d{1,3})|(\[[\da-fA-F:]+\])")


# ── Helpers ───────────────────────────────────────────────────────────────────

def shannon_entropy(s):
    if not s:
        return 0.0
    freq = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    n = len(s)
    return -sum((v / n) * math.log2(v / n) for v in freq.values())


def parse_url(url):
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    return urllib.parse.urlparse(url), url


def check_dns(hostname):
    try:
        socket.setdefaulttimeout(5)
        socket.gethostbyname(hostname)
        return 1
    except Exception:
        return -1


def check_ssl(hostname):
    ctx = ssl.create_default_context()
    try:
        with ctx.wrap_socket(
            socket.create_connection((hostname, 443), timeout=5),
            server_hostname=hostname
        ):
            return 1
    except ssl.SSLCertVerificationError:
        return -1
    except Exception:
        return 0


def fetch_html(url):
    if not REQUESTS_OK:
        return None, 0
    try:
        resp = requests.get(
            url, timeout=10, verify=False, allow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        return resp.text, len(resp.history)
    except Exception:
        return None, 0


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features(url):
    parsed, url = parse_url(url)

    scheme   = parsed.scheme.lower()
    hostname = parsed.hostname or ""   # e.g. "www.google.com"
    port     = str(parsed.port) if parsed.port else ""
    path     = parsed.path
    query    = parsed.query

    # Domain decomposition — keep www so it counts as a subdomain
    parts      = hostname.split(".")
    tld        = ("." + parts[-1])  if len(parts) >= 2 else ""
    sld        = parts[-2]          if len(parts) >= 2 else hostname
    reg_domain = sld + tld          # "google.com"
    subdomains = parts[:-2]         if len(parts) >  2 else []

    full_url_lower = url.lower()

    # ── URL-level features ────────────────────────────────────────────────────
    url_has_ip               = 1 if IP_PATTERN.search(hostname) else 0
    url_length               = len(url)
    url_length_flag          = 1 if url_length > 75 else 0
    uses_url_shortener       = 1 if hostname in URL_SHORTENERS else 0
    url_has_at_symbol        = 1 if "@" in url else 0

    path_query               = path + ("?" + query if query else "")
    url_double_slash_redirect= 1 if "//" in path_query else 0

    url_prefix_suffix_dash   = 1 if "-" in hostname else 0
    url_subdomain_count      = len(subdomains)
    url_subdomain_flag       = 1 if url_subdomain_count >= 2 else 0

    special_chars            = re.findall(r"[!@#$%^&*()+=\[\]{};'\":<>?,\\|`~]",
                                          path + query)
    url_special_char_count   = len(special_chars)
    url_has_suspicious_kw    = 1 if any(k in full_url_lower for k in SUSPICIOUS_KEYWORDS) else 0
    url_dot_count            = url.count(".")

    # ── Domain-level features ─────────────────────────────────────────────────
    # domain_length / domain_entropy computed on registered domain (sld+tld)
    # to match training data computation (e.g. "google.com" not just "google")
    domain_length     = len(reg_domain)
    domain_has_numerics = 1 if any(c.isdigit() for c in sld) else 0
    domain_entropy    = round(shannon_entropy(reg_domain), 4)
    domain_dot_count  = hostname.count(".")
    https_in_domain_token = 1 if "https" in hostname.lower() else 0
    tld_suspicious    = 1 if tld.lower() in SUSPICIOUS_TLDS else 0
    non_standard_port = 1 if port in STANDARD_PORTS else 0

    ssl_final_state         = check_ssl(hostname) if scheme == "https" else 0
    dns_record_exists       = check_dns(hostname)
    domain_reg_length_flag  = -1   # requires WHOIS
    domain_age_flag         = -1   # requires WHOIS
    web_traffic_flag        = -1   # requires external API
    pagerank_flag           = -1   # requires external API
    google_index_flag       = -1   # requires external API
    links_pointing_flag     = -1   # requires external API
    statistical_report_flag = -1   # requires external API

    # ── HTML features ─────────────────────────────────────────────────────────
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

    if REQUESTS_OK and dns_record_exists == 1:
        html_content, html_redirect_count = fetch_html(url)

        if html_content and BS4_OK:
            soup   = BeautifulSoup(html_content, "html.parser")
            html_lc = html_content.lower()

            # favicon
            fav = soup.find("link", rel=lambda r: r and "icon" in " ".join(r).lower())
            if fav and fav.get("href"):
                fav_domain = urllib.parse.urlparse(fav["href"]).hostname or ""
                favicon_external = 0 if (not fav_domain or hostname in fav_domain) else 1
            else:
                favicon_external = 0

            # external resource ratio
            all_res = (soup.find_all("img",    src=True) +
                       soup.find_all("script", src=True) +
                       soup.find_all("link",   href=True))
            ext_res = [r for r in all_res
                       if urllib.parse.urlparse(
                           r.get("src") or r.get("href", "")
                       ).hostname not in ("", None, hostname)]
            html_external_req_ratio = (round(len(ext_res) / len(all_res), 4)
                                       if all_res else 0)

            # anchor ratio
            anchors     = soup.find_all("a", href=True)
            ext_anchors = [a for a in anchors
                           if urllib.parse.urlparse(a["href"]).hostname
                           not in ("", None, hostname)]
            html_anchor_url_ratio = (round(len(ext_anchors) / len(anchors), 4)
                                     if anchors else 0)

            # links / meta / script check
            total_links  = len(soup.find_all("a"))
            total_script = len(soup.find_all("script"))
            html_links_meta_script = 1 if (total_links + total_script) == 0 else 0

            # form actions
            forms = soup.find_all("form")
            html_form_count  = len(forms)
            sfh_susp         = 0
            sub_email        = 0
            for form in forms:
                action = (form.get("action") or "").strip().lower()
                if action in ("", "about:blank") or (
                        urllib.parse.urlparse(action).hostname
                        not in ("", None, hostname)):
                    sfh_susp = 1
                if action.startswith("mailto:"):
                    sub_email = 1
            html_sfh_suspicious   = sfh_susp
            html_submits_to_email = sub_email

            html_abnormal_url         = 0 if hostname in html_lc else 1
            html_status_bar_custom    = 1 if "onmouseover"   in html_lc else 0
            html_disables_right_click = 1 if "contextmenu"   in html_lc else 0
            html_uses_popup           = 1 if "window.open"   in html_lc else 0
            html_uses_iframe          = 1 if soup.find("iframe") else 0

        elif html_content:
            html_lc = html_content.lower()
            html_status_bar_custom    = 1 if "onmouseover" in html_lc else 0
            html_disables_right_click = 1 if "event.button==2" in html_lc else 0
            html_uses_popup           = 1 if "window.open"  in html_lc else 0
            html_uses_iframe          = 1 if "<iframe"       in html_lc else 0
            html_form_count           = html_lc.count("<form")

    # ── Secondary (mirrors of primaries) ──────────────────────────────────────
    sec_domain_age_flag         = domain_age_flag
    sec_dns_record_exists       = dns_record_exists
    sec_web_traffic_flag        = -1
    sec_links_pointing_flag     = -1
    sec_statistical_report_flag = -1
    sec_ssl_state               = ssl_final_state
    sec_tld_suspicious          = tld_suspicious
    sec_has_suspicious_kw       = url_has_suspicious_kw

    return {
        "url_has_ip"                 : url_has_ip,
        "url_length"                 : url_length,
        "url_length_flag"            : url_length_flag,
        "uses_url_shortener"         : uses_url_shortener,
        "url_has_at_symbol"          : url_has_at_symbol,
        "url_double_slash_redirect"  : url_double_slash_redirect,
        "url_prefix_suffix_dash"     : url_prefix_suffix_dash,
        "url_subdomain_count"        : url_subdomain_count,
        "url_subdomain_flag"         : url_subdomain_flag,
        "ssl_final_state"            : ssl_final_state,
        "domain_reg_length_flag"     : domain_reg_length_flag,
        "favicon_external"           : favicon_external,
        "non_standard_port"          : non_standard_port,
        "https_in_domain_token"      : https_in_domain_token,
        "url_special_char_count"     : url_special_char_count,
        "url_has_suspicious_kw"      : url_has_suspicious_kw,
        "url_dot_count"              : url_dot_count,
        "dns_record_exists"          : dns_record_exists,
        "domain_age_flag"            : domain_age_flag,
        "domain_length"              : domain_length,
        "domain_has_numerics"        : domain_has_numerics,
        "tld_suspicious"             : tld_suspicious,
        "web_traffic_flag"           : web_traffic_flag,
        "pagerank_flag"              : pagerank_flag,
        "google_index_flag"          : google_index_flag,
        "links_pointing_flag"        : links_pointing_flag,
        "statistical_report_flag"    : statistical_report_flag,
        "domain_entropy"             : domain_entropy,
        "domain_dot_count"           : domain_dot_count,
        "html_external_req_ratio"    : html_external_req_ratio,
        "html_anchor_url_ratio"      : html_anchor_url_ratio,
        "html_links_meta_script"     : html_links_meta_script,
        "html_sfh_suspicious"        : html_sfh_suspicious,
        "html_submits_to_email"      : html_submits_to_email,
        "html_abnormal_url"          : html_abnormal_url,
        "html_redirect_count"        : html_redirect_count,
        "html_status_bar_custom"     : html_status_bar_custom,
        "html_disables_right_click"  : html_disables_right_click,
        "html_uses_popup"            : html_uses_popup,
        "html_uses_iframe"           : html_uses_iframe,
        "html_form_count"            : html_form_count,
        "sec_domain_age_flag"        : sec_domain_age_flag,
        "sec_dns_record_exists"      : sec_dns_record_exists,
        "sec_web_traffic_flag"       : sec_web_traffic_flag,
        "sec_links_pointing_flag"    : sec_links_pointing_flag,
        "sec_statistical_report_flag": sec_statistical_report_flag,
        "sec_ssl_state"              : sec_ssl_state,
        "sec_tld_suspicious"         : sec_tld_suspicious,
        "sec_has_suspicious_kw"      : sec_has_suspicious_kw,
    }


# ── Prediction ────────────────────────────────────────────────────────────────

def predict(url, model):
    print(f"  Analyzing  : {url}")
    print(f"  Extracting features ...")

    features = extract_features(url)
    df       = pd.DataFrame([features])[FEATURE_ORDER]
    pred     = model.predict(df)[0]
    prob     = model.predict_proba(df)[0]

    label   = "PHISHING"   if pred == 1 else "LEGITIMATE"
    phish_p = prob[1] * 100
    legit_p = prob[0] * 100

    bar_len = 30
    filled  = int(round(phish_p / 100 * bar_len))
    bar     = "#" * filled + "-" * (bar_len - filled)

    print()
    print(f"  Result     : {label}")
    print(f"  Phishing   : {phish_p:5.1f}%  [{bar}]")
    print(f"  Legitimate : {legit_p:5.1f}%")
    print()
    print(f"  Key signals detected:")
    print(f"    URL length          : {features['url_length']}"
          f"{'  (LONG)' if features['url_length_flag'] else ''}")
    print(f"    Has IP in URL       : {'YES (suspicious)' if features['url_has_ip'] else 'no'}")
    print(f"    Has @ symbol        : {'YES (suspicious)' if features['url_has_at_symbol'] else 'no'}")
    print(f"    Uses URL shortener  : {'YES (suspicious)' if features['uses_url_shortener'] else 'no'}")
    print(f"    Prefix/suffix dash  : {'YES (suspicious)' if features['url_prefix_suffix_dash'] else 'no'}")
    print(f"    Suspicious keywords : {'YES' if features['url_has_suspicious_kw'] else 'no'}")
    print(f"    Suspicious TLD      : {'YES' if features['tld_suspicious'] else 'no'}")
    print(f"    DNS resolves        : {'yes' if features['dns_record_exists'] == 1 else 'NO (suspicious)' if features['dns_record_exists'] == -1 else 'unknown'}")
    print(f"    SSL certificate     : {'valid' if features['ssl_final_state'] == 1 else 'no HTTPS' if features['ssl_final_state'] == 0 else 'INVALID (suspicious)'}")
    print(f"    Subdomains          : {features['url_subdomain_count']}")
    print(f"    Domain              : {features['domain_length']} chars, entropy={features['domain_entropy']}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    SEP = "=" * 60
    print(SEP)
    print("  Phishing URL Detector  |  LightGBM Model")
    print(SEP)

    if not REQUESTS_OK:
        print("\n  [!] requests not installed. HTML features will be -1.")
        print("      pip install requests beautifulsoup4\n")
    if REQUESTS_OK and not BS4_OK:
        print("\n  [!] beautifulsoup4 not installed. HTML parsing disabled.")
        print("      pip install beautifulsoup4\n")

    print("\n  Loading model ...")
    model = joblib.load("lightgbm_model.pkl")
    print("  Model loaded.\n")
    print("  Type a URL to check. Type 'quit' to exit.")

    while True:
        print("-" * 60)
        try:
            url = input("  Enter URL: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Exiting.")
            break

        if not url:
            continue
        if url.lower() in ("quit", "exit", "q"):
            print("  Exiting.")
            break

        try:
            predict(url, model)
        except Exception as e:
            print(f"  Error analysing URL: {e}")

    print()


if __name__ == "__main__":
    main()
