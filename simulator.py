"""
metro_sim_human_final.py

Human-feel Delhi Metro simulator (CLI).
Comments and prompts are written in casual/Hinglish tone.

Features implemented per your requests:
1) Greeting message on start
2) Main menu with 3 options: Search Metro, Plan Journey, Exit
3) Search Metro: ask Line, Station, Time -> show Next metro and Subsequent metros
4) Plan Journey: ask Source, Destination, Departure time -> show step-by-step timeline,
   handle interchanges (show where to change), show next connecting train and final arrival
5) Optional: after plan, ask if user wants to provide a deadline to compute "free time" left

Data file expected: metro_data.txt in same folder with CSV lines like:
Line,Station,NextStation,TravelTimeMin,Interchange

This program is intentionally written in a natural, slightly informal style (Hinglish comments)
"""

from math import ceil
import heapq
import sys

DATA_FILE = "delhi_metro.txt"

# service window
SERVICE_START = 6 * 60   # 06:00
SERVICE_END = 23 * 60    # 23:00

PEAK_WINDOWS = [(8*60, 10*60), (17*60, 19*60)]
PEAK_FREQ = 4    # minutes
OFFPEAK_FREQ = 8
INTERCHANGE_DELAY = 5


# ----------------- small helpers -----------------

def to_min(tstr):
    tstr = tstr.strip()
    if ":" not in tstr:
        raise ValueError("Time must be HH:MM")
    h, m = tstr.split(":")
    return int(h) * 60 + int(m)


def to_hhmm(m):
    m = int(m)
    h = (m // 60) % 24
    mm = m % 60
    return f"{h:02d}:{mm:02d}"


def in_service(mins):
    return SERVICE_START <= mins <= SERVICE_END


def freq_at(mins):
    for s, e in PEAK_WINDOWS:
        if s <= mins < e:
            return PEAK_FREQ
    return OFFPEAK_FREQ


# ----------------- data load -----------------

def load_data(filename):
    """Return (edges, station_lines, line_graphs)
    edges: station -> list of (neighbor, time, line)
    line_graphs: line -> {station: [(neighbor, time), ...]}
    station_lines: station -> set(lines)
    """
    edges = {}
    line_graphs = {}
    station_lines = {}
    try:
        with open(filename, encoding='utf-8') as f:
            for raw in f:
                row = raw.strip()
                if not row or row.startswith('#') or row.lower().startswith('line,'):
                    continue
                parts = [p.strip() for p in row.split(',')]
                if len(parts) < 5:
                    continue
                line_name, s1, s2, tstr, _ = parts[:5]
                try:
                    t = int(tstr)
                except ValueError:
                    continue
                edges.setdefault(s1, []).append((s2, t, line_name))
                edges.setdefault(s2, []).append((s1, t, line_name))
                line_graphs.setdefault(line_name, {})
                line_graphs[line_name].setdefault(s1, []).append((s2, t))
                line_graphs[line_name].setdefault(s2, []).append((s1, t))
                station_lines.setdefault(s1, set()).add(line_name)
                station_lines.setdefault(s2, set()).add(line_name)
    except FileNotFoundError:
        raise
    return edges, station_lines, line_graphs


# ----------------- graph utils -----------------

def dijkstra(adj, start):
    dist = {start: 0}
    prev = {}
    pq = [(0, start)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, 10**18):
            continue
        for v, wt in adj.get(u, []):
            nd = d + wt
            if nd < dist.get(v, 10**18):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    return dist, prev


def dijkstra_full(edges, start, goal):
    dist = {start: 0}
    prev = {}
    pq = [(0, start)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, 10**18):
            continue
        if u == goal:
            break
        for v, wt, ln in edges.get(u, []):
            nd = d + wt
            if nd < dist.get(v, 10**18):
                dist[v] = nd
                prev[v] = (u, ln)
                heapq.heappush(pq, (nd, v))
    if goal not in dist:
        return None, []
    # reconstruct
    path = []
    cur = goal
    while cur != start:
        p = prev.get(cur)
        if not p:
            return None, []
        prev_station, line_used = p
        path.append((prev_station, cur, line_used))
        cur = prev_station
    path.reverse()
    return dist[goal], path


# ----------------- offsets (time from termini) -----------------

def find_termini(adj):
    termini = []
    for s, nbrs in adj.items():
        uniq = set(n for n, _ in nbrs)
        if len(uniq) == 1:
            termini.append(s)
    if len(termini) >= 2:
        return termini[:2]
    keys = list(adj.keys())
    if len(keys) >= 2:
        return [keys[0], keys[-1]]
    return keys


def precompute_offsets(line_graphs):
    offsets = {}
    for ln, adj in line_graphs.items():
        terms = find_termini(adj)
        if not terms:
            continue
        a = terms[0]
        b = terms[1] if len(terms) > 1 else a
        dist_a, _ = dijkstra(adj, a)
        dist_b, _ = dijkstra(adj, b)
        offsets[ln] = {"A": (a, dist_a), "B": (b, dist_b)}
    return offsets


# ----------------- timetable helpers -----------------

def next_train_from_terminus(terminus_map, station, now_min, freq):
    if not in_service(now_min):
        return None
    offset = terminus_map.get(station)
    if offset is None:
        return None
    base = SERVICE_START + offset
    if now_min <= base:
        arrival = base
    else:
        k = ceil((now_min - base) / freq)
        arrival = base + k * freq
    if arrival > SERVICE_END:
        return None
    return arrival


def get_next_train_in_direction(line, frm, to, current_min, offsets, freq=None):
    if freq is None:
        freq = freq_at(current_min)
    if line not in offsets:
        return None
    a_term, a_map = offsets[line]["A"]
    b_term, b_map = offsets[line]["B"]
    a_from = a_map.get(frm); a_to = a_map.get(to)
    b_from = b_map.get(frm); b_to = b_map.get(to)
    chosen = None
    if a_from is not None and a_to is not None:
        chosen = a_map if a_to > a_from else b_map
    elif b_from is not None and b_to is not None:
        chosen = b_map if b_to > b_from else a_map
    elif a_from is not None:
        chosen = a_map
    elif b_from is not None:
        chosen = b_map
    else:
        return None
    return next_train_from_terminus(chosen, frm, current_min, freq)


def next_trains_at_station(line, station, current_time_str, offsets, count=5):
    current_min = to_min(current_time_str)
    if not in_service(current_min):
        return []
    if line not in offsets:
        return []
    freq = freq_at(current_min)
    a_term, a_map = offsets[line]["A"]
    b_term, b_map = offsets[line]["B"]
    a_next = next_train_from_terminus(a_map, station, current_min, freq)
    b_next = next_train_from_terminus(b_map, station, current_min, freq)
    results = []
    seeds = []
    if a_next is not None:
        seeds.append((a_next, f"From {a_term}"))
    if b_next is not None:
        seeds.append((b_next, f"From {b_term}"))
    step = 0
    while len(seeds) < count and step < 200:
        step += 1
        if a_next:
            na = a_next + step * freq
            if na <= SERVICE_END:
                seeds.append((na, f"From {a_term}"))
        if b_next:
            nb = b_next + step * freq
            if nb <= SERVICE_END:
                seeds.append((nb, f"From {b_term}"))
    seeds.sort()
    return seeds[:count]


# ----------------- main planner -----------------

def plan_journey(edges, station_lines, line_graphs, offsets, source, dest, start_time_str):
    start_min = to_min(start_time_str)
    if not in_service(start_min):
        return {"error": "No service available at this time (06:00-23:00)."}
    total_time, raw_path = dijkstra_full(edges, source, dest)
    if not raw_path:
        return {"error": f"No path found between {source} and {dest}."}
    stations_seq = [source]
    lines_seq = []
    for u, v, ln in raw_path:
        stations_seq.append(v)
        lines_seq.append(ln)
    timeline = []
    current_time = start_min
    first_line = lines_seq[0]
    next_train = get_next_train_in_direction(first_line, source, stations_seq[1], current_time, offsets, freq=freq_at(current_time))
    if next_train is None:
        return {"error": "No scheduled train available from source at this time."}
    wait = max(0, next_train - current_time)
    timeline.append((source, current_time, None, f"Wait for next {first_line} train at {to_hhmm(next_train)} (wait {wait} min)"))
    current_time = next_train
    prev_line = first_line
    for idx in range(len(lines_seq)):
        u = stations_seq[idx]
        v = stations_seq[idx+1]
        ln = lines_seq[idx]
        travel_time = None
        for nbr, wt, l in edges.get(u, []):
            if nbr == v and l == ln:
                travel_time = wt
                break
        if travel_time is None:
            for nbr, wt, l in edges.get(u, []):
                if nbr == v:
                    travel_time = wt
                    break
        travel_time = travel_time or 0
        arrival = current_time + travel_time
        timeline.append((v, arrival, ln, f"Arrived via {ln} (travel {travel_time} min)"))
        current_time = arrival
        if idx + 1 < len(lines_seq):
            next_ln = lines_seq[idx+1]
            if next_ln != ln:
                # interchange
                timeline.append((v, current_time, None, f"Interchange to {next_ln}, interchange delay {INTERCHANGE_DELAY} min"))
                current_time += INTERCHANGE_DELAY
                next_target = stations_seq[idx+2]
                nxt = get_next_train_in_direction(next_ln, v, next_target, current_time, offsets, freq=freq_at(current_time))
                if nxt is None:
                    return {"error": f"No connecting train found on {next_ln} from {v} after interchange."}
                wait2 = max(0, nxt - current_time)
                timeline.append((v, nxt, next_ln, f"Next {next_ln} train at {to_hhmm(nxt)} (wait {wait2} min)"))
                current_time = nxt
        prev_line = ln
    total_travel = current_time - start_min
    result = {
        "source": source,
        "dest": dest,
        "start_time": start_time_str,
        "end_time": to_hhmm(current_time),
        "total_time_min": total_travel,
        "stations_seq": stations_seq,
        "lines_seq": lines_seq,
        "timeline": timeline
    }
    return result


# ----------------- CLI & run helpers -----------------

def show_welcome():
    print("\n=== Shubh yatra: Delhi Metro Simulator ===")
    print("Welcome! Main aapki madad karunga metro timings aur routes plan karne mein.")


def option_search(offsets, edges):
    print("\n-- Search Metro --")
    line = input("Line (exact name): ").strip()
    station = input("Station (exact name): ").strip()
    time = input("Time you want to catch train (HH:MM): ").strip()
    try:
        _ = to_min(time)
    except Exception:
        print("Bad time format. Use HH:MM")
        return
    if line not in offsets:
        print("Line not found in data.")
        return
    upcoming = next_trains_at_station(line, station, time, offsets, count=6)
    if not upcoming:
        print("No upcoming trains (maybe outside service or station/line mismatch).")
        return
    # show next and subsequent in required format
    next_one = upcoming[0][0]
    subsequent = [to_hhmm(t) for t, _ in upcoming[1:]]
    print(f"\nNext metro at {to_hhmm(next_one)}")
    if subsequent:
        print("Subsequent metros at " + ", ".join(subsequent) + ", ...")
    else:
        print("No further metros found for today.")


def option_plan(edges, station_lines, line_graphs, offsets):
    print("\n-- Plan Journey --")
    src = input("Source station (exact): ").strip()
    dst = input("Destination station (exact): ").strip()
    start_time = input("When do you want to start (HH:MM): ").strip()
    try:
        _ = to_min(start_time)
    except Exception:
        print("Bad time format.")
        return
    plan = plan_journey(edges, station_lines, line_graphs, offsets, src, dst, start_time)
    if "error" in plan:
        print("Error:", plan["error"])
        return
    print("\nJourney Plan:")
    # human-friendly summary
    print(f"Start at {plan['source']}")
    # timeline lines
    for station, tm, ln, note in plan['timeline']:
        tstr = to_hhmm(tm)
        if ln:
            print(f" {tstr} - {station} (via {ln}) - {note}")
        else:
            print(f" {tstr} - {station} - {note}")
    print(f"Total travel time: {plan['total_time_min']} minutes")
    # optional: ask for deadline to compute free time
    choice = input("\nDo you want to provide a deadline to compute free time after arrival? (y/n): ").strip().lower()
    if choice == 'y':
        dl = input("Enter required-arrival time (HH:MM): ").strip()
        try:
            dl_min = to_min(dl)
            arrival_min = to_min(plan['end_time'])
            if dl_min >= arrival_min:
                free = dl_min - arrival_min
                print(f"You will have {free} minutes free before your deadline.")
            else:
                late_by = arrival_min - dl_min
                print(f"You will be late by {late_by} minutes relative to deadline.")
        except Exception:
            print("Bad time format for deadline.")


def main():
    show_welcome()
    # load data
    try:
        edges, station_lines, line_graphs = load_data(DATA_FILE)
    except FileNotFoundError:
        print("metro_data.txt not found in current folder. Please add data and rerun.")
        return
    offsets = precompute_offsets(line_graphs)
    print(f"Data loaded. Stations: {len(edges)}; Lines: {len(line_graphs)}")

    while True:
        print("\nChoose an option:")
        print(" 1) Search Metro")
        print(" 2) Plan Journey")
        print(" 3) Exit")
        ch = input("Enter choice (1/2/3): ").strip()
        if ch == '1':
            option_search(offsets, edges)
        elif ch == '2':
            option_plan(edges, station_lines, line_graphs, offsets)
        elif ch == '3':
            print("Goodbye — safe journey!")
            break
        else:
            print("Invalid choice, choose 1, 2 or 3.")


if __name__ == '__main__':
    main()
