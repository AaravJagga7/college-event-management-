"""
================================================================================
RV College of Engineering (RVCE) — 8th Mile Festival Dataset Generator
================================================================================
Faithfully grounded in the official RVCE 8th Mile Festival Timetable:
  • Real Venues: Main Stage, IEM Auditorium, Civil Sem. Hall, ECE Sem.-Hall,
                 ISE Sem. Hall, ET Sem. Hall, MCA & BT Sem. Hall, Chem Sem. Hall,
                 ME Sem. Hall, DTL Huddle, Video Room, Memorial Lane,
                 Civil Dept Stretch, NCC Training Ground, Biotech Quad,
                 CV-114 Classroom, ET-101/102 Classrooms, Classrooms.
  • Official 3-Day 8th Mile Timeline Events:
      - Day 1: Kannada Habba, Rangataranga, Theme Photography, Amazing Race,
               Mr. & Ms. 8th Mile, Name It, Fandom Quiz: Anime, Pranesh Stand-Up,
               Raghu Vinestore, Sid 18 Rapper, Naveen Sajju, Overnight Hackathon,
               Satellite Payload Design.
      - Day 2: Shaurya, CS:GO/Mortal Kombat/FIFA, F1 Simulator, Stock Market,
               Math-O-Mania, Soul Train, Euphorism, Mr. & Ms. 8th Mile R2,
               Fandom Quiz: TV & Movies, Open Mic, Sketch Showdown, Amaal Mallik Pro-Night,
               Business Marathon.
      - Day 3: Cquence, Aircrash Investigation, Mock IPL Auction Quiz, Clash of Chords,
               Morse Decoding, RC Car Rukus, Mock G20 Summit, 5D Space Venture,
               TVS Stunt Show, Business Marathon Final, Verve Fashion, Mock IPL Auction,
               Progressive Brothers EDM Night.
================================================================================
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

TOTAL_ROWS = 1500

# ------------------------------------------------------------------------------
# 1. RVCE VENUES & CAPACITY SPECIFICATIONS
# ------------------------------------------------------------------------------
VENUES = [
    {"name": "Main Stage", "room_no": 100, "capacity": 1500, "weight": 0.12},
    {"name": "IEM Auditorium", "room_no": 101, "capacity": 350, "weight": 0.10},
    {"name": "Civil Sem. Hall", "room_no": 102, "capacity": 160, "weight": 0.08},
    {"name": "ECE Sem.-Hall", "room_no": 103, "capacity": 180, "weight": 0.08},
    {"name": "ISE Sem. Hall", "room_no": 104, "capacity": 160, "weight": 0.08},
    {"name": "ET Sem. Hall", "room_no": 105, "capacity": 140, "weight": 0.06},
    {"name": "MCA & BT Sem. Hall", "room_no": 106, "capacity": 150, "weight": 0.07},
    {"name": "Chem Sem. Hall", "room_no": 107, "capacity": 120, "weight": 0.05},
    {"name": "ME Sem. Hall", "room_no": 108, "capacity": 130, "weight": 0.05},
    {"name": "BT Quadrangle", "room_no": 109, "capacity": 300, "weight": 0.05},
    {"name": "DTL Huddle", "room_no": 111, "capacity": 120, "weight": 0.05},
    {"name": "Video Room", "room_no": 112, "capacity": 60, "weight": 0.03},
    {"name": "CV-114 Classroom", "room_no": 114, "capacity": 60, "weight": 0.03},
    {"name": "Memorial Lane", "room_no": 115, "capacity": 250, "weight": 0.03},
    {"name": "Civil Dept Stretch", "room_no": 116, "capacity": 600, "weight": 0.03},
    {"name": "NCC Training Ground", "room_no": 117, "capacity": 500, "weight": 0.03},
    {"name": "ET-101 / ET-102 Classrooms", "room_no": 118, "capacity": 80, "weight": 0.03},
    {"name": "ISE Classroom", "room_no": 201, "capacity": 70, "weight": 0.03},
    {"name": "ECE Classroom", "room_no": 202, "capacity": 70, "weight": 0.03},
    {"name": "ASE Classroom", "room_no": 203, "capacity": 70, "weight": 0.02},
    {"name": "BT Classroom", "room_no": 204, "capacity": 70, "weight": 0.02},
    {"name": "Chem Classroom", "room_no": 205, "capacity": 70, "weight": 0.02},
]

# ------------------------------------------------------------------------------
# 2. RVCE 8th MILE EVENTS TAXONOMY (OFFICIAL SCHEDULE)
# ------------------------------------------------------------------------------
EVENTS_CATALOG = [
    # --- DAY 1 OFFICIAL EVENTS ---
    {"event": "Kannada Habba", "cat": "cultural", "type": "Very important", "pref_venue": "Main Stage", "base_turnout": 850, "base_cost": 65000, "def_time": "9:00am", "def_day": 1},
    {"event": "Rangataranga", "cat": "cultural", "type": "Important", "pref_venue": "IEM Auditorium", "base_turnout": 280, "base_cost": 28000, "def_time": "11:00am", "def_day": 1},
    {"event": "Theme Based Photography", "cat": "fun_club", "type": "Normal", "pref_venue": "Video Room", "base_turnout": 45, "base_cost": 8000, "def_time": "1:00pm", "def_day": 1},
    {"event": "Amazing Race", "cat": "fun_club", "type": "Important", "pref_venue": "ISE Sem. Hall", "base_turnout": 130, "base_cost": 16000, "def_time": "2:00pm", "def_day": 1},
    {"event": "Mr. & Ms. 8th Mile", "cat": "cultural", "type": "Very important", "pref_venue": "ECE Sem.-Hall", "base_turnout": 160, "base_cost": 32000, "def_time": "2:00pm", "def_day": 1},
    {"event": "Name It", "cat": "academic", "type": "Normal", "pref_venue": "Chem Sem. Hall", "base_turnout": 80, "base_cost": 10000, "def_time": "2:30pm", "def_day": 1},
    {"event": "Fandom Quiz: Anime", "cat": "fun_club", "type": "Important", "pref_venue": "MCA & BT Sem. Hall", "base_turnout": 120, "base_cost": 14000, "def_time": "3:00pm", "def_day": 1},
    {"event": "Pranesh (Stand-up)", "cat": "cultural", "type": "Very important", "pref_venue": "Main Stage", "base_turnout": 950, "base_cost": 75000, "def_time": "3:00pm", "def_day": 1},
    {"event": "Raghu Vinestore", "cat": "cultural", "type": "Very important", "pref_venue": "Main Stage", "base_turnout": 1100, "base_cost": 85000, "def_time": "7:00pm", "def_day": 1},
    {"event": "Sid 18 (Rapper)", "cat": "cultural", "type": "Very important", "pref_venue": "Main Stage", "base_turnout": 1150, "base_cost": 88000, "def_time": "7:30pm", "def_day": 1},
    {"event": "Naveen Sajju", "cat": "cultural", "type": "Very important", "pref_venue": "Main Stage", "base_turnout": 1250, "base_cost": 95000, "def_time": "8:00pm", "def_day": 1},
    {"event": "Overnight Hackathon", "cat": "technical", "type": "Very important", "pref_venue": "DTL Huddle", "base_turnout": 110, "base_cost": 35000, "def_time": "10:00pm", "def_day": 1},
    {"event": "Satellite Payload Design", "cat": "technical", "type": "Important", "pref_venue": "IEM Auditorium", "base_turnout": 180, "base_cost": 25000, "def_time": "10:00pm", "def_day": 1},

    # --- DAY 2 OFFICIAL EVENTS ---
    {"event": "Shaurya (Day 2)", "cat": "sports", "type": "Important", "pref_venue": "NCC Training Ground", "base_turnout": 220, "base_cost": 30000, "def_time": "9:00am", "def_day": 2},
    {"event": "CS:GO, Mortal Kombat, FIFA", "cat": "technical", "type": "Important", "pref_venue": "ECE Classroom", "base_turnout": 65, "base_cost": 18000, "def_time": "9:00am", "def_day": 2},
    {"event": "F1 Simulator", "cat": "technical", "type": "Important", "pref_venue": "CV-114 Classroom", "base_turnout": 55, "base_cost": 22000, "def_time": "9:00am", "def_day": 2},
    {"event": "Stock Market Competition", "cat": "academic", "type": "Important", "pref_venue": "Civil Sem. Hall", "base_turnout": 135, "base_cost": 20000, "def_time": "10:00am", "def_day": 2},
    {"event": "Math-O-Mania", "cat": "academic", "type": "Normal", "pref_venue": "ET-101 / ET-102 Classrooms", "base_turnout": 65, "base_cost": 10000, "def_time": "10:00am", "def_day": 2},
    {"event": "Soul Train", "cat": "cultural", "type": "Very important", "pref_venue": "Main Stage", "base_turnout": 500, "base_cost": 40000, "def_time": "10:00am", "def_day": 2},
    {"event": "Euphorism", "cat": "cultural", "type": "Important", "pref_venue": "ET Sem. Hall", "base_turnout": 120, "base_cost": 20000, "def_time": "10:00am", "def_day": 2},
    {"event": "Mr. & Ms. 8th Mile Round 2", "cat": "cultural", "type": "Very important", "pref_venue": "MCA & BT Sem. Hall", "base_turnout": 140, "base_cost": 28000, "def_time": "12:00pm", "def_day": 2},
    {"event": "Fandom Quiz: TV & Movies", "cat": "fun_club", "type": "Normal", "pref_venue": "MCA & BT Sem. Hall", "base_turnout": 95, "base_cost": 11000, "def_time": "2:15pm", "def_day": 2},
    {"event": "Open Mic", "cat": "cultural", "type": "Important", "pref_venue": "Main Stage", "base_turnout": 350, "base_cost": 22000, "def_time": "3:00pm", "def_day": 2},
    {"event": "Sketch Showdown", "cat": "cultural", "type": "Normal", "pref_venue": "DTL Huddle", "base_turnout": 70, "base_cost": 12000, "def_time": "4:00pm", "def_day": 2},
    {"event": "Amaal Mallik Live Concert", "cat": "cultural", "type": "Very important", "pref_venue": "Main Stage", "base_turnout": 1450, "base_cost": 120000, "def_time": "8:00pm", "def_day": 2},
    {"event": "Business Marathon", "cat": "academic", "type": "Important", "pref_venue": "DTL Huddle", "base_turnout": 100, "base_cost": 24000, "def_time": "10:00pm", "def_day": 2},

    # --- DAY 3 OFFICIAL EVENTS ---
    {"event": "Cquence", "cat": "technical", "type": "Important", "pref_venue": "Civil Sem. Hall", "base_turnout": 110, "base_cost": 18000, "def_time": "10:00am", "def_day": 3},
    {"event": "Aircrash Investigation", "cat": "academic", "type": "Important", "pref_venue": "ECE Sem.-Hall", "base_turnout": 120, "base_cost": 20000, "def_time": "10:00am", "def_day": 3},
    {"event": "Mock IPL Auction Quiz", "cat": "sports", "type": "Important", "pref_venue": "MCA & BT Sem. Hall", "base_turnout": 130, "base_cost": 18000, "def_time": "10:00am", "def_day": 3},
    {"event": "Clash of Chords", "cat": "cultural", "type": "Very important", "pref_venue": "Main Stage", "base_turnout": 600, "base_cost": 50000, "def_time": "10:00am", "def_day": 3},
    {"event": "Morse Decoding", "cat": "technical", "type": "Normal", "pref_venue": "ME Sem. Hall", "base_turnout": 65, "base_cost": 10000, "def_time": "10:00am", "def_day": 3},
    {"event": "RC Car Rukus", "cat": "technical", "type": "Important", "pref_venue": "Memorial Lane", "base_turnout": 180, "base_cost": 26000, "def_time": "10:00am", "def_day": 3},
    {"event": "Mock G20 Summit", "cat": "academic", "type": "Very important", "pref_venue": "DTL Huddle", "base_turnout": 100, "base_cost": 30000, "def_time": "10:00am", "def_day": 3},
    {"event": "5D Space Venture", "cat": "technical", "type": "Important", "pref_venue": "BT Quadrangle", "base_turnout": 220, "base_cost": 32000, "def_time": "11:00am", "def_day": 3},
    {"event": "TVS Stunt Show", "cat": "sports", "type": "Very important", "pref_venue": "Civil Dept Stretch", "base_turnout": 550, "base_cost": 60000, "def_time": "12:00pm", "def_day": 3},
    {"event": "Business Marathon: Final", "cat": "academic", "type": "Important", "pref_venue": "IEM Auditorium", "base_turnout": 220, "base_cost": 28000, "def_time": "12:00pm", "def_day": 3},
    {"event": "Verve, Mr. & Ms. 8th Mile", "cat": "cultural", "type": "Very important", "pref_venue": "Main Stage", "base_turnout": 950, "base_cost": 65000, "def_time": "2:30pm", "def_day": 3},
    {"event": "Mock IPL Auction", "cat": "sports", "type": "Important", "pref_venue": "Civil Sem. Hall", "base_turnout": 150, "base_cost": 25000, "def_time": "3:00pm", "def_day": 3},
    {"event": "Progressive Brothers", "cat": "cultural", "type": "Very important", "pref_venue": "Main Stage", "base_turnout": 1450, "base_cost": 115000, "def_time": "8:00pm", "def_day": 3},
    
    # Campus club & department preliminaries
    {"event": "Mystere.exe (Cryptic Hunt)", "cat": "technical", "type": "Important", "pref_venue": "ECE Classroom", "base_turnout": 65, "base_cost": 16000, "def_time": "4:00pm", "def_day": 1},
    {"event": "Cloud Pirates (DevOps Hack)", "cat": "technical", "type": "Important", "pref_venue": "ISE Sem. Hall", "base_turnout": 120, "base_cost": 22000, "def_time": "11:00am", "def_day": 2},
    {"event": "Aero Ops (Drone & CAD Flight)", "cat": "technical", "type": "Important", "pref_venue": "ASE Classroom", "base_turnout": 60, "base_cost": 18000, "def_time": "2:00pm", "def_day": 3},
    {"event": "Vasuki Vaibhav Live Concert", "cat": "cultural", "type": "Very important", "pref_venue": "Main Stage", "base_turnout": 1200, "base_cost": 95000, "def_time": "7:00pm", "def_day": 1},
    {"event": "DJ Night (Lights & Bass)", "cat": "cultural", "type": "Very important", "pref_venue": "Main Stage", "base_turnout": 1300, "base_cost": 85000, "def_time": "8:00pm", "def_day": 2},
]

# ------------------------------------------------------------------------------
# 3. RVCE FACULTY LEADERS
# ------------------------------------------------------------------------------
FACULTY_LIST = [
    {"id": "RVCE_FAC0101", "name": "Dr. Shanmukha N", "desig": "Professor", "dept": "Mechanical"},
    {"id": "RVCE_FAC0102", "name": "Dr. Ramesh K", "desig": "Professor", "dept": "Civil"},
    {"id": "RVCE_FAC0103", "name": "Dr. Geetha V", "desig": "Professor", "dept": "ISE"},
    {"id": "RVCE_FAC0104", "name": "Dr. K.N. Subramanya", "desig": "Professor", "dept": "IEM"},
    {"id": "RVCE_FAC0105", "name": "Dr. Renuka Prasad", "desig": "Professor", "dept": "MCA"},
    {"id": "RVCE_FAC0106", "name": "Dr. Vidya Niranjan", "desig": "Professor", "dept": "Biotechnology"},
    {"id": "RVCE_FAC0107", "name": "Dr. Ravish Aradhya", "desig": "Professor", "dept": "ECE"},
    {"id": "RVCE_FAC0108", "name": "Dr. Mamatha G.S", "desig": "Associate Professor", "dept": "ISE"},
    {"id": "RVCE_FAC0109", "name": "Prof. Anand Kumar", "desig": "Associate Professor", "dept": "ECE"},
    {"id": "RVCE_FAC0110", "name": "Prof. Sunitha B", "desig": "Associate Professor", "dept": "Chemical"},
    {"id": "RVCE_FAC0111", "name": "Prof. Chethana R", "desig": "Associate Professor", "dept": "Aerospace"},
    {"id": "RVCE_FAC0112", "name": "Dr. K. Viswanath", "desig": "Associate Professor", "dept": "ETE"},
    {"id": "RVCE_FAC0113", "name": "Dr. Preethi N", "desig": "Associate Professor", "dept": "CSE"},
    {"id": "RVCE_FAC0114", "name": "Prof. Harish Kumar", "desig": "Assistant Professor", "dept": "Mathematics"},
    {"id": "RVCE_FAC0115", "name": "Prof. Prakash Biswagar", "desig": "Assistant Professor", "dept": "Cultural Committee"},
    {"id": "RVCE_FAC0116", "name": "Prof. Girish Rao", "desig": "Assistant Professor", "dept": "Sports Dept"},
    {"id": "RVCE_FAC0117", "name": "Prof. Shwetha M", "desig": "Assistant Professor", "dept": "ISE"},
    {"id": "RVCE_FAC0118", "name": "Prof. Abhishek Gowda", "desig": "Assistant Professor", "dept": "ECE"},
    {"id": "RVCE_FAC0119", "name": "Prof. Deepa K", "desig": "Assistant Professor", "dept": "CSE"},
    {"id": "RVCE_FAC0120", "name": "Prof. Naveen Chandra", "desig": "Assistant Professor", "dept": "Mechanical"},
]

venue_lookup = {v["name"]: v for v in VENUES}
venue_names = [v["name"] for v in VENUES]
venue_weights = np.array([v["weight"] for v in VENUES], dtype=float)
venue_weights = venue_weights / venue_weights.sum()

TIME_SLOTS = ["9:00am", "9:30am", "10:00am", "10:30am", "11:00am", "12:00pm", "1:00pm", "2:00pm", "2:30pm", "3:00pm", "4:00pm", "7:00pm", "7:30pm", "8:00pm", "10:00pm"]
DURATIONS = ["60min", "90min", "120min", "180min", "240min"]
DURATION_WEIGHTS = [0.20, 0.40, 0.25, 0.10, 0.05]

def parse_time_min(t_str):
    dt = datetime.strptime(t_str, "%I:%M%p")
    return dt.hour * 60 + dt.minute

occupied_intervals = {}
records = []

# ------------------------------------------------------------------------------
# 4. INSERT OFFICIAL TIMELINE RECORDS (Day 1: 06/15/2023, Day 2: 06/16/2023, Day 3: 06/17/2023)
# ------------------------------------------------------------------------------
official_festival_days = {
    1: datetime(2023, 6, 15),
    2: datetime(2023, 6, 16),
    3: datetime(2023, 6, 17)
}

sno = 1
for ev_item in EVENTS_CATALOG:
    day_num = ev_item.get("def_day", 1)
    req_date = official_festival_days[day_num]
    lead_days = np.random.randint(5, 20)
    submitted_date = req_date - timedelta(days=lead_days)
    time_slot = ev_item.get("def_time", "10:00am")
    dur_str = "120min" if ev_item["type"] == "Very important" else "90min"
    dur_min = int(dur_str.replace("min", ""))

    v_name = ev_item["pref_venue"]
    chosen_venue = venue_lookup.get(v_name, venue_lookup["Main Stage"])
    fac = FACULTY_LIST[np.random.choice(len(FACULTY_LIST))]

    cap = chosen_venue["capacity"]
    base_t = ev_item["base_turnout"]
    registers = int(np.clip(base_t + np.random.normal(0, 15), 20, cap))

    t_start = parse_time_min(time_slot)
    t_end = t_start + dur_min
    room_date_key = (chosen_venue["room_no"], req_date.strftime("%m/%d/%Y"))

    has_clash = any(t_start < e and s < t_end for (s, e) in occupied_intervals.get(room_date_key, []))

    accept_reject = "Reject" if (registers > cap or has_clash) else "Accept"
    room_avail = "Not Available" if has_clash else "Available"
    reject_reason = "Room conflict" if has_clash else ("Capacity exceeded" if registers > cap else "")

    if accept_reject == "Accept":
        occupied_intervals.setdefault(room_date_key, []).append((t_start, t_end))

    cat_base = {"cultural": 24000, "technical": 18000, "sports": 15000, "academic": 11000, "fun_club": 7000}[ev_item["cat"]]
    type_mult = {"Normal": 1.0, "Important": 1.5, "Very important": 2.2}[ev_item["type"]]
    desig_bonus = {"Professor": 7000, "Associate Professor": 4000, "Assistant Professor": 1500}[fac["desig"]]
    total_budget = int(np.round((cat_base * type_mult + cap * 130 + registers * 190 + (dur_min / 60) * 4000 + desig_bonus) / 100) * 100)

    records.append({
        "SNO": sno,
        "Event": ev_item["event"],
        "Event_Category": ev_item["cat"],
        "Faculty_id": fac["id"],
        "Faculty_name": fac["name"],
        "Faculty_Designation": fac["desig"],
        "Registers": registers,
        "Type": ev_item["type"],
        "Request for Booking": 1,
        "Request for Cancelling": 0,
        "Requested Room No": chosen_venue["room_no"],
        "Requested Venue": chosen_venue["name"],
        "Room_Capacity": cap,
        "Request Date": req_date.strftime("%m/%d/%Y"),
        "Request Submitted On": submitted_date.strftime("%m/%d/%Y"),
        "Requested Time": time_slot,
        "Duration": dur_str,
        "Room Available/Not Available": room_avail,
        "Accept/Reject": accept_reject,
        "Reject_Reason": reject_reason,
        "Budget": total_budget
    })
    sno += 1

# ------------------------------------------------------------------------------
# 5. GENERATE COMPREHENSIVE SEMESTER & FESTIVAL WORKSHOP RUNS UP TO 1,500 ROWS
# ------------------------------------------------------------------------------
while sno <= TOTAL_ROWS:
    ev_item = EVENTS_CATALOG[np.random.choice(len(EVENTS_CATALOG))]
    
    if np.random.rand() < 0.70 and ev_item["pref_venue"] in venue_lookup:
        chosen_venue = venue_lookup[ev_item["pref_venue"]]
    else:
        chosen_venue = venue_lookup[np.random.choice(venue_names, p=venue_weights)]
        
    fac = FACULTY_LIST[np.random.choice(len(FACULTY_LIST))]
    
    month_choice = np.random.choice([6, 9, 10, 11, 12], p=[0.20, 0.20, 0.22, 0.22, 0.16])
    if month_choice == 6:
        day = np.random.choice([14, 15, 16, 17, 18, 19])
        year = 2023
    else:
        day = np.random.randint(1, 28)
        year = 2025
        
    req_date = datetime(year, month_choice, day)
    lead_days = int(np.random.exponential(scale=6)) + 2
    lead_days = min(max(lead_days, 1), 30)
    submitted_date = req_date - timedelta(days=lead_days)
    
    time_slot = np.random.choice(TIME_SLOTS)
    dur_str = np.random.choice(DURATIONS, p=DURATION_WEIGHTS)
    dur_min = int(dur_str.replace("min", ""))
    
    cap = chosen_venue["capacity"]
    base_t = ev_item["base_turnout"]
    
    # 4% controlled overcapacity rate for policy checks
    if np.random.rand() < 0.045:
        registers = int(cap * np.random.uniform(1.08, 1.35))
    else:
        fill_target = np.random.uniform(0.40, 0.95)
        registers = int(np.clip(cap * fill_target + np.random.normal(0, 8), 15, cap))
        
    t_start = parse_time_min(time_slot)
    t_end = t_start + dur_min
    room_date_key = (chosen_venue["room_no"], req_date.strftime("%m/%d/%Y"))
    
    has_clash = any(t_start < e and s < t_end for (s, e) in occupied_intervals.get(room_date_key, []))
    cap_exceeded = registers > cap
    
    if has_clash:
        accept_reject = "Reject"
        reject_reason = "Room conflict"
        room_avail = "Not Available"
    elif cap_exceeded:
        accept_reject = "Reject"
        reject_reason = "Capacity exceeded"
        room_avail = "Available"
    elif np.random.rand() < 0.015:
        accept_reject = "Reject"
        reject_reason = "Policy/other"
        room_avail = "Available"
    else:
        accept_reject = "Accept"
        reject_reason = ""
        room_avail = "Available"
        occupied_intervals.setdefault(room_date_key, []).append((t_start, t_end))
        
    cat_base = {
        "cultural": 24000,
        "technical": 18000,
        "sports": 15000,
        "academic": 11000,
        "fun_club": 7000
    }[ev_item["cat"]]
    
    type_mult = {"Normal": 1.0, "Important": 1.5, "Very important": 2.2}[ev_item["type"]]
    desig_bonus = {"Professor": 7000, "Associate Professor": 4000, "Assistant Professor": 1500}[fac["desig"]]
    rush_mod = 4500 if lead_days < 4 else (-2000 if lead_days >= 14 else 0)
    
    venue_scale_cost = cap * 130
    turnout_cost = registers * 190
    duration_cost = (dur_min / 60.0) * 4000
    noise_budget = np.random.normal(0, 2800)
    
    total_budget = (cat_base * type_mult) + venue_scale_cost + turnout_cost + duration_cost + desig_bonus + rush_mod + noise_budget
    total_budget = int(np.round(np.maximum(total_budget, 6000) / 100) * 100)
    
    records.append({
        "SNO": sno,
        "Event": ev_item["event"],
        "Event_Category": ev_item["cat"],
        "Faculty_id": fac["id"],
        "Faculty_name": fac["name"],
        "Faculty_Designation": fac["desig"],
        "Registers": registers,
        "Type": ev_item["type"],
        "Request for Booking": 1,
        "Request for Cancelling": 0,
        "Requested Room No": chosen_venue["room_no"],
        "Requested Venue": chosen_venue["name"],
        "Room_Capacity": cap,
        "Request Date": req_date.strftime("%m/%d/%Y"),
        "Request Submitted On": submitted_date.strftime("%m/%d/%Y"),
        "Requested Time": time_slot,
        "Duration": dur_str,
        "Room Available/Not Available": room_avail,
        "Accept/Reject": accept_reject,
        "Reject_Reason": reject_reason,
        "Budget": total_budget
    })
    sno += 1

df_rvce = pd.DataFrame(records)

df_rvce.to_csv("rvce_8th_mile_data.csv", index=False)
df_rvce.to_csv("synthetic_event_data.csv", index=False)

print(f"🎉 Successfully generated {len(df_rvce)} rows for RVCE 8th Mile Festival!")
print("Saved files:")
print("  • rvce_8th_mile_data.csv")
print("  • synthetic_event_data.csv")
print("\n--- DATASET SUMMARY ---")
print(f"Events:       {df_rvce['Event'].nunique()} distinct 8th Mile events")
print(f"Venues:       {df_rvce['Requested Venue'].nunique()} RVCE campus venues")
print(f"Approval:     {df_rvce['Accept/Reject'].value_counts(normalize=True).round(3).to_dict()}")
print(f"Attendance:   Mean={df_rvce['Registers'].mean():.1f}, Max={df_rvce['Registers'].max()}")
print(f"Budget (₹):   Mean=₹{df_rvce['Budget'].mean():,.2f}, Max=₹{df_rvce['Budget'].max():,.2f}")
