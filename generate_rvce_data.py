"""
================================================================================
RV College of Engineering (RVCE) — 8th Mile Festival Dataset Generator
================================================================================
Generates 1,200 realistic event booking, attendance, and budgeting records
faithfully grounded in the RVCE 8th Mile / Ashtrang Festival Timetable:
  • Real Venues: Main Stage, IEM Auditorium, Civil Sem. Hall, ECE Sem.-Hall,
                 ISE Sem. Hall, ET Sem. Hall, MCA Sem. Hall, Chem Sem. Hall,
                 Old Lib., BT Quadrangle, ECE Labs, Classrooms (ISE, ECE, ASE, BT, Chem).
  • Real Events: Clash of Chords, Verve, Alaap Performance, Soul Train,
                 Ranga-Parva, Yakshagana, Vasuki Vaibhav Live, DJ Night,
                 Pro Night, Mr. & Ms. 8th Mile, Flare, Mystere.exe,
                 Cloud Pirates, Aero Ops, Inno-Venture, BGMI, Valorant, FIFA,
                 Fandom Quizz, Math O Mania & Pictoral, Murder Under the Microscope,
                 Business Marathon, Mock IPL, Shutter-Bug, Canvas Craze,
                 Cinephile Circle, Stand Up Comedy, Shaurya, etc.
  • Real RVCE Faculty & Department Coordinators.
================================================================================
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

TOTAL_ROWS = 1200

# ------------------------------------------------------------------------------
# 1. RVCE VENUES & CAPACITY SPECIFICATIONS
# ------------------------------------------------------------------------------
VENUES = [
    {"name": "Main Stage", "room_no": 100, "capacity": 1500, "weight": 0.08},
    {"name": "IEM Auditorium", "room_no": 101, "capacity": 350, "weight": 0.12},
    {"name": "Civil Sem. Hall", "room_no": 102, "capacity": 160, "weight": 0.10},
    {"name": "ECE Sem.-Hall", "room_no": 103, "capacity": 180, "weight": 0.10},
    {"name": "ISE Sem. Hall", "room_no": 104, "capacity": 160, "weight": 0.09},
    {"name": "ET Sem. Hall", "room_no": 105, "capacity": 140, "weight": 0.08},
    {"name": "MCA Sem. Hall", "room_no": 106, "capacity": 130, "weight": 0.07},
    {"name": "Chem Sem. Hall", "room_no": 107, "capacity": 120, "weight": 0.06},
    {"name": "Old Lib.", "room_no": 108, "capacity": 120, "weight": 0.06},
    {"name": "BT Quadrangle", "room_no": 109, "capacity": 300, "weight": 0.06},
    {"name": "ECE Labs", "room_no": 110, "capacity": 80, "weight": 0.05},
    {"name": "ISE Classroom", "room_no": 201, "capacity": 70, "weight": 0.04},
    {"name": "ECE Classroom", "room_no": 202, "capacity": 70, "weight": 0.04},
    {"name": "ASE Classroom", "room_no": 203, "capacity": 70, "weight": 0.04},
    {"name": "BT Classroom", "room_no": 204, "capacity": 70, "weight": 0.04},
    {"name": "Chem Classroom", "room_no": 205, "capacity": 70, "weight": 0.03},
]

# ------------------------------------------------------------------------------
# 2. RVCE 8th MILE EVENTS TAXONOMY
# ------------------------------------------------------------------------------
EVENTS_CATALOG = [
    # Cultural (Music, Dance, Theatre, Concerts)
    {"event": "Clash of Chords", "cat": "cultural", "type": "Very important", "pref_venue": "Main Stage", "base_turnout": 450, "base_cost": 45000},
    {"event": "Verve (Fashion & Dance)", "cat": "cultural", "type": "Very important", "pref_venue": "Main Stage", "base_turnout": 500, "base_cost": 50000},
    {"event": "Alaap Performance (Music Club)", "cat": "cultural", "type": "Important", "pref_venue": "Main Stage", "base_turnout": 350, "base_cost": 25000},
    {"event": "Soul Train", "cat": "cultural", "type": "Very important", "pref_venue": "Main Stage", "base_turnout": 400, "base_cost": 35000},
    {"event": "Ranga-Parva (Kannada Theatre)", "cat": "cultural", "type": "Important", "pref_venue": "IEM Auditorium", "base_turnout": 220, "base_cost": 22000},
    {"event": "Yakshagana Heritage Dance", "cat": "cultural", "type": "Very important", "pref_venue": "Main Stage", "base_turnout": 380, "base_cost": 30000},
    {"event": "Vasuki Vaibhav Live Concert", "cat": "cultural", "type": "Very important", "pref_venue": "Main Stage", "base_turnout": 1200, "base_cost": 95000},
    {"event": "DJ Night (Lights & Bass)", "cat": "cultural", "type": "Very important", "pref_venue": "Main Stage", "base_turnout": 1300, "base_cost": 85000},
    {"event": "Pro Night (Star Concert)", "cat": "cultural", "type": "Very important", "pref_venue": "Main Stage", "base_turnout": 1400, "base_cost": 110000},
    {"event": "Mr. & Ms. 8th Mile", "cat": "cultural", "type": "Very important", "pref_venue": "ET Sem. Hall", "base_turnout": 130, "base_cost": 28000},
    {"event": "Flare (Choreo Dance)", "cat": "cultural", "type": "Important", "pref_venue": "BT Quadrangle", "base_turnout": 240, "base_cost": 24000},

    # Technical (Hackathons, Coding, Robotics, Gaming)
    {"event": "Mystere.exe (Cryptic Hunt)", "cat": "technical", "type": "Important", "pref_venue": "ECE Classroom", "base_turnout": 65, "base_cost": 16000},
    {"event": "Cloud Pirates (DevOps Hack)", "cat": "technical", "type": "Important", "pref_venue": "ISE Sem. Hall", "base_turnout": 120, "base_cost": 22000},
    {"event": "Aero Ops (Drone & CAD Flight)", "cat": "technical", "type": "Important", "pref_venue": "ASE Classroom", "base_turnout": 60, "base_cost": 18000},
    {"event": "Inno-Venture (Hardware Expo)", "cat": "technical", "type": "Very important", "pref_venue": "ECE Sem.-Hall", "base_turnout": 140, "base_cost": 26000},
    {"event": "BGMI Esports Championship", "cat": "technical", "type": "Important", "pref_venue": "ECE Sem.-Hall", "base_turnout": 150, "base_cost": 20000},
    {"event": "Valorant Tournament", "cat": "technical", "type": "Important", "pref_venue": "ECE Labs", "base_turnout": 75, "base_cost": 17000},
    {"event": "FIFA 25 Console League", "cat": "technical", "type": "Normal", "pref_venue": "ASE Classroom", "base_turnout": 60, "base_cost": 12000},

    # Academic & Strategic Intellectual
    {"event": "Fandom Quizz", "cat": "academic", "type": "Important", "pref_venue": "IEM Auditorium", "base_turnout": 200, "base_cost": 15000},
    {"event": "Math O Mania & Pictoral", "cat": "academic", "type": "Normal", "pref_venue": "ISE Classroom", "base_turnout": 55, "base_cost": 9000},
    {"event": "Murder Under the Microscope", "cat": "academic", "type": "Important", "pref_venue": "BT Classroom", "base_turnout": 65, "base_cost": 14000},
    {"event": "Business Marathon", "cat": "academic", "type": "Very important", "pref_venue": "Civil Sem. Hall", "base_turnout": 130, "base_cost": 25000},
    {"event": "Mock IPL Auction", "cat": "academic", "type": "Important", "pref_venue": "Civil Sem. Hall", "base_turnout": 140, "base_cost": 18000},

    # Fun Club & Creative Arts
    {"event": "Shutter-Bug Photography", "cat": "fun_club", "type": "Normal", "pref_venue": "MCA Sem. Hall", "base_turnout": 80, "base_cost": 10000},
    {"event": "Canvas Craze (Live Painting)", "cat": "fun_club", "type": "Normal", "pref_venue": "ECE Classroom", "base_turnout": 50, "base_cost": 8500},
    {"event": "Cinephile Circle", "cat": "fun_club", "type": "Normal", "pref_venue": "Civil Sem. Hall", "base_turnout": 90, "base_cost": 11000},
    {"event": "Stand Up Comedy Showcase", "cat": "fun_club", "type": "Important", "pref_venue": "IEM Auditorium", "base_turnout": 220, "base_cost": 20000},
    {"event": "Procession & Inaugural Rally", "cat": "fun_club", "type": "Very important", "pref_venue": "Main Stage", "base_turnout": 500, "base_cost": 35000},
    {"event": "Cquence Storyboard", "cat": "fun_club", "type": "Normal", "pref_venue": "Civil Sem. Hall", "base_turnout": 70, "base_cost": 9500},

    # Sports & Physical Challenges
    {"event": "Shaurya Endurance Arena", "cat": "sports", "type": "Important", "pref_venue": "Old Lib.", "base_turnout": 110, "base_cost": 22000},
    {"event": "Box Cricket Knockout", "cat": "sports", "type": "Normal", "pref_venue": "BT Quadrangle", "base_turnout": 120, "base_cost": 14000},
    {"event": "Gully Football Cup", "cat": "sports", "type": "Normal", "pref_venue": "BT Quadrangle", "base_turnout": 110, "base_cost": 13000},
]

# ------------------------------------------------------------------------------
# 3. RVCE FACULTY REPOSITORIES
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

# Time slots & Durations
TIME_SLOTS = ["9:00am", "9:30am", "10:00am", "10:30am", "11:00am", "1:00pm", "1:30pm", "2:00pm", "3:00pm", "4:00pm", "7:00pm", "9:00pm"]
DURATIONS = ["60min", "90min", "120min", "180min", "240min"]
DURATION_WEIGHTS = [0.15, 0.40, 0.30, 0.10, 0.05]

# ------------------------------------------------------------------------------
# 4. GENERATE 1,200 ROWS WITH TIMETABLE COHERENCE
# ------------------------------------------------------------------------------
records = []
start_date = datetime(2025, 9, 1)

# Chronological tracker for room conflicts
occupied_intervals = {} # (room_no, date_str) -> list of (start_min, end_min)

def parse_time_min(t_str):
    dt = datetime.strptime(t_str, "%I:%M%p")
    return dt.hour * 60 + dt.minute

for i in range(1, TOTAL_ROWS + 1):
    ev_item = EVENTS_CATALOG[np.random.choice(len(EVENTS_CATALOG))]
    
    # 70% chance to book preferred venue, 30% chance another venue
    if np.random.rand() < 0.70 and ev_item["pref_venue"] in venue_lookup:
        chosen_venue = venue_lookup[ev_item["pref_venue"]]
    else:
        chosen_venue = venue_lookup[np.random.choice(venue_names, p=venue_weights)]
        
    fac = FACULTY_LIST[np.random.choice(len(FACULTY_LIST))]
    
    # Scheduling dates across the semester (Sept 1 to Dec 20, 2025)
    # Reflects club workshops, department qualifiers, and 8th Mile mega fest
    month_choice = np.random.choice([9, 10, 11, 12], p=[0.24, 0.28, 0.30, 0.18])
    if month_choice == 9:
        day = np.random.randint(1, 31)
    elif month_choice == 10:
        day = np.random.randint(1, 32)
    elif month_choice == 11:
        day = np.random.randint(1, 31)
    else:
        # Dec 4, 5, 6 are 8th Mile core festival days
        if np.random.rand() < 0.25:
            day = np.random.choice([4, 5, 6])
        else:
            day = np.random.randint(1, 24)
            
    req_date = datetime(2025, month_choice, day)
    lead_days = int(np.random.exponential(scale=6)) + 2
    lead_days = min(max(lead_days, 1), 30)
    submitted_date = req_date - timedelta(days=lead_days)
    
    time_slot = np.random.choice(TIME_SLOTS)
    dur_str = np.random.choice(DURATIONS, p=DURATION_WEIGHTS)
    dur_min = int(dur_str.replace("min", ""))
    
    # Student Turnout estimation
    cap = chosen_venue["capacity"]
    base_t = ev_item["base_turnout"]
    
    # Generate realistic registrations centered safely within venue capacity
    # with a controlled 4% rate of accidental over-capacity requests
    is_overcapacity_candidate = (np.random.rand() < 0.045)
    if is_overcapacity_candidate:
        registers = int(cap * np.random.uniform(1.08, 1.35))
    else:
        # Turnout typically occupies 40% - 95% of venue capacity
        fill_target = np.random.uniform(0.40, 0.95)
        registers = int(np.clip(cap * fill_target + np.random.normal(0, 8), 15, cap))
    
    # Spatio-temporal room clash detection
    t_start = parse_time_min(time_slot)
    t_end = t_start + dur_min
    room_date_key = (chosen_venue["room_no"], req_date.strftime("%m/%d/%Y"))
    
    has_clash = False
    for (s, e) in occupied_intervals.get(room_date_key, []):
        if t_start < e and s < t_end:
            has_clash = True
            break
            
    # Decision logic
    cap_exceeded = registers > cap
    
    if has_clash:
        accept_reject = "Reject"
        reject_reason = "Room conflict"
        room_avail = "Not Available"
    elif cap_exceeded:
        accept_reject = "Reject"
        reject_reason = "Capacity exceeded"
        room_avail = "Available"
    elif np.random.rand() < 0.015: # rare administrative policy reject
        accept_reject = "Reject"
        reject_reason = "Policy/other"
        room_avail = "Available"
    else:
        accept_reject = "Accept"
        reject_reason = ""
        room_avail = "Available"
        occupied_intervals.setdefault(room_date_key, []).append((t_start, t_end))
        
    # Budget Calculation (INR)
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
        "SNO": i,
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

df_rvce = pd.DataFrame(records)

# Save to rvce_8th_mile_data.csv and synthetic_event_data.csv
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
print("\nFirst 5 rows:")
print(df_rvce[["Event", "Requested Venue", "Registers", "Room_Capacity", "Accept/Reject", "Budget"]].head())
