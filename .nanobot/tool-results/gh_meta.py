import json, urllib.request

def get(url):
    req = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "nanobot-research",
    })
    with urllib.request.urlopen(req, timeout=25) as r:
        return json.load(r)

print("=== search: image segmentation (top by stars) ===")
d = get("https://api.github.com/search/repositories?q=image+segmentation+archived:false&sort=stars&order=desc&per_page=15")
for r in d.get("items", []):
    lic = (r.get("license") or {}).get("spdx_id", "?")
    print(f"{r['full_name']:45} stars={r['stargazers_count']:>7} lang={str(r.get('language')):>12} lic={lic}")

print()
print("=== chosen candidates ===")
for repo in ["facebookresearch/sam2", "ultralytics/ultralytics", "open-mmlab/mmsegmentation",
             "MIC-DKFZ/nnUNet", "facebookresearch/detectron2"]:
    r = get(f"https://api.github.com/repos/{repo}")
    lic = (r.get("license") or {}).get("spdx_id", "?")
    desc = (r.get("description") or "").strip()
    print(f"{r['full_name']:40} | stars={r['stargazers_count']:>7} | forks={r['forks_count']:>6} | "
          f"lang={str(r.get('language')):>12} | lic={lic:>12} | branch={r.get('default_branch')} | "
          f"pushed={r.get('pushed_at','?')[:10]}")
    print(f"    topics: {r.get('topics', [])[:8]}")
    print(f"    desc: {desc[:160]}")
