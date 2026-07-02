import sys, os, subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SCRAPERS = [
    ('Kufar', 'src/scrape_kufar.py'),
    ('Realt', 'src/scrape_realt.py'),
]


def run_scraper(name, script_path):
    if not os.path.exists(script_path):
        print(f'[SKIP] {name}: script not found at {script_path}')
        return False

    print(f'[RUN] {name}: python {script_path}...')
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=True, text=True, timeout=600,
    )

    print(result.stdout)

    if result.returncode != 0:
        print(f'[FAIL] {name}: exit code {result.returncode}')
        if result.stderr:
            print(f'  stderr: {result.stderr[:500]}')
        return False

    print(f'[OK] {name} done')
    return True


def rebuild_timeseries():
    print('\n[BUILD] Rebuilding timeseries data...')
    from src.build_timeseries import build_timeseries
    build_timeseries()


def main():
    all_ok = True
    for name, path in SCRAPERS:
        ok = run_scraper(name, path)
        if not ok:
            all_ok = False

    print()
    if all_ok:
        rebuild_timeseries()
    else:
        print('[WARN] Some scrapers failed, timeseries not rebuilt')

    print('\n=== Done ===')


if __name__ == '__main__':
    main()
