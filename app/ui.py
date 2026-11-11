"""
================================================================================
RVCE 8th Mile Event Intelligence & MLOps Platform - Interactive Web UI
================================================================================
Clean, executive dashboard matching the official student event management design:
  - Top KPI Summary Cards with double-circle teal icons & subtext
  - Registrations Timeline Bar Chart with toolbar controls
  - Years of Graduation & Account Types Distribution Donut Charts with callout labels
  - Live "Switch to Text View" toggle
  - Working "Copy" buttons with clipboard toasts
  - Integrated Real-Time Triple-Task AI Predictor & Kubeflow Model Telemetry
================================================================================
"""

UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>RVCE 8th Mile — AI Event Intelligence Dashboard</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    body {
      font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background-color: #f0f2f5;
      color: #334155;
    }
    
    .font-mono { font-family: 'JetBrains Mono', monospace; }
    
    /* Clean Card Styling matching uploaded design */
    .dashboard-card {
      background: #ffffff;
      border: 1px solid #e2e8f0;
      border-radius: 8px;
      box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
      transition: all 0.2s ease;
    }
    .dashboard-card:hover {
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);
    }
    
    /* Circular Icon Badge matching screenshot double-ring */
    .icon-badge {
      width: 58px;
      height: 58px;
      border-radius: 50%;
      border: 2px solid #5eead4;
      padding: 3px;
      display: flex;
      align-items: center;
      justify-content: center;
      background: #f0fdfa;
      flex-shrink: 0;
    }
    .icon-badge-inner {
      width: 100%;
      height: 100%;
      border-radius: 50%;
      border: 1.5px solid #2dd4bf;
      display: flex;
      align-items: center;
      justify-content: center;
      color: #0d9488;
      font-size: 20px;
    }

    /* Olive-green pill button from screenshot */
    .btn-olive {
      background-color: #557a2b;
      color: #ffffff;
      font-size: 13px;
      font-weight: 600;
      padding: 6px 14px;
      border-radius: 4px;
      transition: background-color 0.15s ease;
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }
    .btn-olive:hover {
      background-color: #476724;
    }

    /* Royal Blue Copy button from screenshot */
    .btn-copy {
      background-color: #0284c7;
      color: #ffffff;
      font-size: 12px;
      font-weight: 600;
      padding: 4px 14px;
      border-radius: 4px;
      transition: background-color 0.15s ease;
    }
    .btn-copy:hover {
      background-color: #0369a1;
    }

    /* Toast Notification */
    #toast {
      visibility: hidden;
      min-width: 260px;
      background-color: #1e293b;
      color: #fff;
      text-align: center;
      border-radius: 6px;
      padding: 12px 18px;
      position: fixed;
      z-index: 100;
      right: 30px;
      bottom: 30px;
      font-size: 14px;
      box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    #toast.show {
      visibility: visible;
      animation: fadein 0.3s, fadeout 0.5s 2.5s;
    }
    @keyframes fadein { from { bottom: 10px; opacity: 0; } to { bottom: 30px; opacity: 1; } }
    @keyframes fadeout { from { bottom: 30px; opacity: 1; } to { bottom: 10px; opacity: 0; } }
  </style>
</head>
<body class="p-4 md:p-6 lg:p-8">

  <!-- ======================================================================== -->
  <!-- MAIN DASHBOARD CONTAINER -->
  <!-- ======================================================================== -->
  <div class="max-w-[1400px] mx-auto bg-white rounded-xl shadow-sm border border-slate-200 p-5 md:p-6">

    <!-- Top Title & Controls Header (Exact Match to Screenshot) -->
    <div class="border-b border-slate-100 pb-4 mb-5 flex flex-wrap items-center justify-between gap-3">
      <div>
        <h1 class="text-sm md:text-base font-semibold text-slate-700">
          Dashboard <span class="font-normal text-slate-500">(RVCE 8th Mile Festival &amp; New Student Orientation — Wed, Jan 1, 2025 8:00 AM – Mon, May 31, 2027 1:00 PM)</span>
        </h1>
        <div class="mt-2.5 flex flex-wrap items-center gap-2">
          <button onclick="toggleTextView()" id="toggleViewBtn" class="btn-olive">
            <i class="fa-solid fa-list-check text-xs"></i>
            <span>Switch to Text View</span>
          </button>
          
          <button onclick="togglePredictorModal()" class="text-xs font-semibold px-3 py-1.5 bg-slate-100 hover:bg-slate-200 text-slate-700 rounded border border-slate-300 transition flex items-center gap-1.5">
            <i class="fa-solid fa-wand-magic-sparkles text-teal-600"></i>
            <span>Live AI Proposal Simulator</span>
          </button>

          <a href="/docs" target="_blank" class="text-xs font-semibold px-3 py-1.5 bg-slate-100 hover:bg-slate-200 text-slate-700 rounded border border-slate-300 transition flex items-center gap-1.5">
            <i class="fa-solid fa-code text-teal-600"></i>
            <span>API Docs</span>
          </a>
        </div>
      </div>

      <!-- Live Service Telemetry Status Badge -->
      <div class="flex items-center gap-2 bg-slate-50 border border-slate-200 px-3 py-2 rounded-lg text-xs">
        <span class="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse"></span>
        <span class="text-slate-600">Kubeflow Registry: <strong class="text-emerald-700">Live (v1alpha3)</strong></span>
        <span class="text-slate-300">|</span>
        <span class="text-slate-500">P95 SLA:</span>
        <span class="font-mono font-bold text-teal-700" id="liveP95">16.4ms</span>
      </div>
    </div>

    <!-- ======================================================================== -->
    <!-- GRAPHICAL VIEW (DEFAULT) -->
    <!-- ======================================================================== -->
    <div id="graphicalView" class="space-y-6">

      <!-- 1. TOP 5 KPI SUMMARY METRIC TILES (EXACT ROW MATCH TO SCREENSHOT) -->
      <div class="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
        
        <!-- Tile 1: UNIQUE USERS -->
        <div class="dashboard-card p-4 flex items-center justify-between">
          <div class="icon-badge">
            <div class="icon-badge-inner">
              <i class="fa-solid fa-id-card"></i>
            </div>
          </div>
          <div class="text-right">
            <div class="text-[11px] font-bold tracking-wider text-slate-500 uppercase">UNIQUE USERS</div>
            <div class="text-2xl md:text-3xl font-extrabold text-slate-800 mt-0.5">14</div>
            <div class="text-[11px] text-slate-400 mt-1">12 Faculty Leads</div>
          </div>
        </div>

        <!-- Tile 2: REGISTRATIONS -->
        <div class="dashboard-card p-4 flex items-center justify-between">
          <div class="icon-badge">
            <div class="icon-badge-inner">
              <i class="fa-solid fa-ticket-simple"></i>
            </div>
          </div>
          <div class="text-right">
            <div class="text-[11px] font-bold tracking-wider text-slate-500 uppercase">REGISTRATIONS</div>
            <div class="text-2xl md:text-3xl font-extrabold text-slate-800 mt-0.5">11</div>
            <div class="text-[11px] text-slate-400 mt-1">14 Unique Users</div>
          </div>
        </div>

        <!-- Tile 3: CHECKED IN -->
        <div class="dashboard-card p-4 flex items-center justify-between">
          <div class="icon-badge">
            <div class="icon-badge-inner">
              <i class="fa-solid fa-check"></i>
            </div>
          </div>
          <div class="text-right">
            <div class="text-[11px] font-bold tracking-wider text-slate-500 uppercase">CHECKED IN</div>
            <div class="text-2xl md:text-3xl font-extrabold text-slate-800 mt-0.5">12</div>
            <div class="text-[11px] text-slate-400 mt-1">Out of 11 Registrations</div>
          </div>
        </div>

        <!-- Tile 4: CHECKED OUT -->
        <div class="dashboard-card p-4 flex items-center justify-between">
          <div class="icon-badge">
            <div class="icon-badge-inner">
              <i class="fa-solid fa-arrow-right-from-bracket"></i>
            </div>
          </div>
          <div class="text-right">
            <div class="text-[11px] font-bold tracking-wider text-slate-500 uppercase">CHECKED OUT</div>
            <div class="text-2xl md:text-3xl font-extrabold text-slate-800 mt-0.5">2</div>
            <div class="text-[11px] text-slate-400 mt-1">Out of 11 Registrations</div>
          </div>
        </div>

        <!-- Tile 5: NO SHOW -->
        <div class="dashboard-card p-4 flex items-center justify-between">
          <div class="icon-badge">
            <div class="icon-badge-inner">
              <i class="fa-solid fa-eye-slash"></i>
            </div>
          </div>
          <div class="text-right">
            <div class="text-[11px] font-bold tracking-wider text-slate-500 uppercase">NO SHOW</div>
            <div class="text-2xl md:text-3xl font-extrabold text-slate-800 mt-0.5">1</div>
            <div class="text-[11px] text-slate-400 mt-1">Out of 11 Registrations</div>
          </div>
        </div>

      </div>

      <!-- 2. THREE ANALYTICS CARDS (EXACT MATCH TO LOWER HALF OF SCREENSHOT) -->
      <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">

        <!-- Card 1: Registrations Timeline Bar Chart -->
        <div class="dashboard-card p-5 flex flex-col justify-between">
          <div>
            <div class="flex items-center justify-between mb-1">
              <h2 class="text-base font-bold text-slate-800">Registrations</h2>
              <button onclick="copyCardData('Registrations', '11 total bookings over 10 active registration days across Jan 28, Feb 24, May 19, Sep 4')" class="btn-copy">
                Copy
              </button>
            </div>
            
            <p class="text-xs text-slate-500 flex items-center gap-1.5 mb-4">
              Registrations took place over
              <span class="inline-flex items-center gap-1 font-semibold text-emerald-700 bg-emerald-50 px-1.5 py-0.5 rounded border border-emerald-200">
                <i class="fa-regular fa-calendar text-[11px]"></i> 10 days
              </span>
            </p>
          </div>

          <!-- Bar Chart Container -->
          <div class="relative w-full h-[220px]">
            <canvas id="registrationsBarChart"></canvas>
            <!-- Chart mini toolbar from screenshot -->
            <div class="absolute top-1 right-1 flex flex-col gap-1 bg-white/90 p-1 rounded shadow-sm border border-slate-200 text-[10px] text-slate-400">
              <button title="Pan" class="hover:text-slate-700 p-0.5"><i class="fa-solid fa-arrows-up-down-left-right"></i></button>
              <button title="Zoom" class="hover:text-slate-700 p-0.5"><i class="fa-solid fa-magnifying-glass-plus"></i></button>
              <button title="Reset" class="hover:text-slate-700 p-0.5"><i class="fa-solid fa-house"></i></button>
            </div>
          </div>

          <div class="pt-3 border-t border-slate-100 flex items-center justify-between text-[11px] text-slate-400">
            <span>Peak Day: <strong class="text-slate-700">May 19 (Main Stage)</strong></span>
            <span>Turnout: <strong class="text-teal-600">92% Capacity</strong></span>
          </div>
        </div>

        <!-- Card 2: Years Of Graduation Donut Chart -->
        <div class="dashboard-card p-5 flex flex-col justify-between">
          <div>
            <div class="flex items-center justify-between mb-1">
              <h2 class="text-base font-bold text-slate-800">Years Of Graduation</h2>
              <button onclick="copyCardData('Years Of Graduation', 'Attendees represented 4 Years of Graduation: 2027 (10%), 2025 (10%), 2022 (15%), Unknown/Other (65%)')" class="btn-copy">
                Copy
              </button>
            </div>
            
            <p class="text-xs text-slate-500 flex items-center gap-1.5 mb-4">
              Your attendees represented
              <span class="inline-flex items-center gap-1 font-semibold text-emerald-700 bg-emerald-50 px-1.5 py-0.5 rounded border border-emerald-200">
                <i class="fa-regular fa-calendar text-[11px]"></i> 4 Years of Graduation
              </span>
            </p>
          </div>

          <!-- Pie / Donut Chart Container -->
          <div class="relative w-full h-[220px] flex items-center justify-center">
            <canvas id="graduationDonutChart"></canvas>
          </div>

          <div class="pt-3 border-t border-slate-100 flex flex-wrap items-center justify-between text-[11px] text-slate-500 gap-1">
            <span class="flex items-center gap-1"><span class="w-2.5 h-2.5 rounded-full bg-[#4ecdc4]"></span> Unknown (65%)</span>
            <span class="flex items-center gap-1"><span class="w-2.5 h-2.5 rounded-full bg-[#ced6e0]"></span> 2022 (15%)</span>
            <span class="flex items-center gap-1"><span class="w-2.5 h-2.5 rounded-full bg-[#2f3542]"></span> 2025 (10%)</span>
            <span class="flex items-center gap-1"><span class="w-2.5 h-2.5 rounded-full bg-[#ff4757]"></span> 2027 (10%)</span>
          </div>
        </div>

        <!-- Card 3: Account Types Donut Chart -->
        <div class="dashboard-card p-5 flex flex-col justify-between">
          <div>
            <div class="flex items-center justify-between mb-1">
              <h2 class="text-base font-bold text-slate-800">Account Types</h2>
              <button onclick="copyCardData('Account Types', 'Attendees represented 4 Account Types: Undergraduate Students (55%), Graduate Students (20%), Faculty/Staff (15%), External/Parents (10%)')" class="btn-copy">
                Copy
              </button>
            </div>
            
            <p class="text-xs text-slate-500 flex items-center gap-1.5 mb-4">
              Your attendees represented
              <span class="inline-flex items-center gap-1 font-semibold text-emerald-700 bg-emerald-50 px-1.5 py-0.5 rounded border border-emerald-200">
                <i class="fa-solid fa-user-group text-[11px]"></i> 4 Account Types
              </span>
            </p>
          </div>

          <!-- Pie / Donut Chart Container -->
          <div class="relative w-full h-[220px] flex items-center justify-center">
            <canvas id="accountTypesDonutChart"></canvas>
          </div>

          <div class="pt-3 border-t border-slate-100 flex flex-wrap items-center justify-between text-[11px] text-slate-500 gap-1">
            <span class="flex items-center gap-1"><span class="w-2.5 h-2.5 rounded-full bg-[#4ecdc4]"></span> Undergrad (55%)</span>
            <span class="flex items-center gap-1"><span class="w-2.5 h-2.5 rounded-full bg-[#ff4757]"></span> Graduate (20%)</span>
            <span class="flex items-center gap-1"><span class="w-2.5 h-2.5 rounded-full bg-[#2f3542]"></span> Faculty/Staff (15%)</span>
            <span class="flex items-center gap-1"><span class="w-2.5 h-2.5 rounded-full bg-[#ced6e0]"></span> Parent/Guest (10%)</span>
          </div>
        </div>

      </div>

      <!-- 3. FESTIVAL MLOPS & MODEL REGISTRY SUMMARY BANNER -->
      <div class="bg-gradient-to-r from-slate-900 to-teal-950 text-white rounded-lg p-4 md:p-5 flex flex-wrap items-center justify-between gap-4">
        <div>
          <div class="flex items-center gap-2">
            <span class="px-2 py-0.5 rounded bg-teal-500/20 text-teal-300 font-mono text-xs font-semibold border border-teal-500/30">
              KUBEFLOW MODEL REGISTRY
            </span>
            <span class="text-xs text-slate-300">Spec Version: v1alpha3</span>
          </div>
          <h3 class="text-sm md:text-base font-bold text-white mt-1">
            Triple-Task Champion Suite: Attendance, Booking Approval &amp; Budget Optimization
          </h3>
          <p class="text-xs text-slate-300 mt-0.5">
            100% Quality Gates Passed (45/45 Unit, Risk, Behavioral &amp; Drift Tests Green).
          </p>
        </div>
        
        <div class="flex items-center gap-3">
          <button onclick="togglePredictorModal()" class="px-4 py-2 bg-teal-500 hover:bg-teal-400 text-slate-950 font-bold text-xs rounded-lg shadow-sm transition">
            <i class="fa-solid fa-play mr-1.5"></i> Test Event Proposal
          </button>
          <a href="/registry/kubeflow" target="_blank" class="px-3.5 py-2 bg-slate-800 hover:bg-slate-700 text-white text-xs font-semibold rounded-lg border border-slate-700 transition">
            View KFMR JSON
          </a>
        </div>
      </div>

    </div>

    <!-- ======================================================================== -->
    <!-- TEXT VIEW (TOGGLED VIA OLIVE BUTTON) -->
    <!-- ======================================================================== -->
    <div id="textView" class="hidden space-y-6">
      <div class="border border-slate-200 rounded-lg p-5 bg-slate-50 font-mono text-xs text-slate-800 leading-relaxed overflow-x-auto">
        <div class="flex items-center justify-between border-b border-slate-200 pb-3 mb-4">
          <span class="font-bold text-slate-900 text-sm">=== EXECUTIVE TEXT AUDIT SUMMARY (RVCE 8TH MILE) ===</span>
          <button onclick="copyTextSummary()" class="btn-copy">Copy All Text</button>
        </div>
        <pre id="rawTextContent">
DASHBOARD AUDIT SPECIFICATION:
================================================================================
Event:             RVCE 8th Mile Festival &amp; New Student Orientation
Time Window:       Wed, Jan 1, 2025 8:00 AM – Mon, May 31, 2027 1:00 PM
Environment:       Production (FastAPI + Kubeflow Model Registry + Airflow)

KEY PERFORMANCE INDICATORS (KPIs):
--------------------------------------------------------------------------------
• Unique Users:                     14
• Total Registrations / Bookings:   11 (14 Unique Users represented)
• Checked In (Approved):            12 (Out of 11 Registrations)
• Checked Out (Room Conflicts):     2  (Out of 11 Registrations)
• No Show (Overcapacity Suppressed):1  (Out of 11 Registrations)

EVENT DURATION &amp; REGISTRATION ACTIVITY:
--------------------------------------------------------------------------------
• Active Registration Period:       10 Days
• Peak Traffic Dates:               January 28, February 24, May 19, September 4
• Highest Turnout Hub:              Main Stage / IEM Auditorium (Capacity: 350-500)

GRADUATION YEAR DISTRIBUTION (ATTENDEES):
--------------------------------------------------------------------------------
• Batch 2027:                       10%
• Batch 2025:                       10%
• Batch 2022:                       15%
• Unknown / General Campus:         65%

ACCOUNT &amp; ATTENDEE ROLE PROFILE:
--------------------------------------------------------------------------------
• Undergraduate Students:           55%
• Graduate / PG Students:           20%
• Faculty &amp; Club Coordinators:      15%
• Parent / External Guests:         10%

ML MODEL VALIDATION &amp; DRIFT BENCHMARKS:
--------------------------------------------------------------------------------
• Attendance Regressor (Task A):    R² = 0.8573 | MAE = 83.15 students
• Approval Classifier (Task B):     ROC-AUC = 0.8694 | F1 = 0.9183 | Acc = 86.0%
• Budget Regressor (Task C):        R² = 0.9991 | MAE = ₹3,686.96 | MAPE = 3.63%
• P95 Single-Sample Inference SLA:  16.4 ms (Threshold <= 25.0 ms)
• Population Stability Index (PSI): room_capacity = 0.0228 (STABLE, < 0.25)
• Disparate Impact (Four-Fifths):   Academic (0.95), Cultural (0.91), Tech (0.93)
================================================================================
        </pre>
      </div>
    </div>

  </div>

  <!-- ======================================================================== -->
  <!-- MODAL: LIVE AI PROPOSAL SIMULATOR -->
  <!-- ======================================================================== -->
  <div id="predictorModal" class="hidden fixed inset-0 z-50 bg-slate-900/60 backdrop-blur-sm flex items-center justify-center p-4">
    <div class="bg-white rounded-xl shadow-2xl max-w-2xl w-full border border-slate-200 overflow-hidden">
      <div class="px-5 py-4 border-b border-slate-100 flex items-center justify-between bg-slate-50">
        <h3 class="font-bold text-slate-800 text-sm md:text-base flex items-center gap-2">
          <i class="fa-solid fa-wand-magic-sparkles text-teal-600"></i>
          Test Live Festival Proposal (Instant AI Inference)
        </h3>
        <button onclick="togglePredictorModal()" class="text-slate-400 hover:text-slate-600">
          <i class="fa-solid fa-xmark text-lg"></i>
        </button>
      </div>

      <div class="p-5 space-y-4 max-h-[75vh] overflow-y-auto text-xs text-slate-700">
        <!-- Preset quick buttons -->
        <div>
          <label class="font-bold text-slate-600 block mb-1.5">Scenario Presets:</label>
          <div class="flex flex-wrap gap-1.5">
            <button onclick="loadScenario('ashtrang')" class="px-2.5 py-1 bg-teal-50 hover:bg-teal-100 text-teal-800 rounded border border-teal-200 font-medium">Ashtrang Cultural Night</button>
            <button onclick="loadScenario('hackathon')" class="px-2.5 py-1 bg-sky-50 hover:bg-sky-100 text-sky-800 rounded border border-sky-200 font-medium">24hr Tech Hackathon</button>
            <button onclick="loadScenario('clash')" class="px-2.5 py-1 bg-rose-50 hover:bg-rose-100 text-rose-800 rounded border border-rose-200 font-medium">High-Clash Risk Venue</button>
          </div>
        </div>

        <form id="simForm" onsubmit="runLiveSimulation(event)" class="grid grid-cols-1 sm:grid-cols-2 gap-3.5 pt-2">
          <div>
            <label class="block font-semibold mb-1">Event Category</label>
            <select id="inpCategory" class="w-full border border-slate-300 rounded p-1.5 bg-white text-xs">
              <option value="cultural">Cultural Festival</option>
              <option value="technical">Technical Workshop / Hackathon</option>
              <option value="academic">Academic Lecture</option>
              <option value="sports">Inter-College Sports</option>
              <option value="fun_club">Fun Club Gaming</option>
            </select>
          </div>

          <div>
            <label class="block font-semibold mb-1">Room Capacity</label>
            <input type="number" id="inpCapacity" value="350" min="50" max="2500" class="w-full border border-slate-300 rounded p-1.5 text-xs">
          </div>

          <div>
            <label class="block font-semibold mb-1">Duration (Minutes)</label>
            <input type="number" id="inpDuration" value="120" min="30" max="480" class="w-full border border-slate-300 rounded p-1.5 text-xs">
          </div>

          <div>
            <label class="block font-semibold mb-1">Lead Time (Days)</label>
            <input type="number" id="inpLeadDays" value="14" min="0" max="180" class="w-full border border-slate-300 rounded p-1.5 text-xs">
          </div>

          <div>
            <label class="block font-semibold mb-1">Earlier Room Clashes</label>
            <input type="number" id="inpClashes" value="0" min="0" max="10" class="w-full border border-slate-300 rounded p-1.5 text-xs">
          </div>

          <div>
            <label class="block font-semibold mb-1">Faculty Rank</label>
            <select id="inpFaculty" class="w-full border border-slate-300 rounded p-1.5 bg-white text-xs">
              <option value="Professor">Professor (Head)</option>
              <option value="Associate Professor">Associate Professor</option>
              <option value="Assistant Professor">Assistant Professor</option>
            </select>
          </div>

          <div class="sm:col-span-2 pt-2">
            <button type="submit" class="w-full py-2 bg-teal-600 hover:bg-teal-700 text-white font-bold rounded shadow transition">
              Run Real-Time Prediction
            </button>
          </div>
        </form>

        <!-- Prediction Result Output Box -->
        <div id="simResult" class="hidden mt-4 p-4 rounded-lg border border-slate-200 bg-slate-50 space-y-2">
          <div class="font-bold text-slate-800 text-sm flex items-center justify-between border-b border-slate-200 pb-2">
            <span>Triple-Task Prediction Results:</span>
            <span id="resBadge" class="px-2 py-0.5 rounded text-[11px] font-bold"></span>
          </div>
          <div class="grid grid-cols-3 gap-2 text-center pt-1 font-mono">
            <div class="p-2 bg-white rounded border border-slate-200">
              <div class="text-[10px] text-slate-400 uppercase font-sans">Predicted Turnout</div>
              <div class="text-base font-bold text-slate-800 mt-0.5" id="resAttendance">--</div>
            </div>
            <div class="p-2 bg-white rounded border border-slate-200">
              <div class="text-[10px] text-slate-400 uppercase font-sans">Approval Probability</div>
              <div class="text-base font-bold text-slate-800 mt-0.5" id="resProb">--</div>
            </div>
            <div class="p-2 bg-white rounded border border-slate-200">
              <div class="text-[10px] text-slate-400 uppercase font-sans">Allocated Budget</div>
              <div class="text-base font-bold text-slate-800 mt-0.5" id="resBudget">--</div>
            </div>
          </div>
        </div>

      </div>
    </div>
  </div>

  <!-- Toast Notification Container -->
  <div id="toast">Copied to clipboard!</div>

  <!-- ======================================================================== -->
  <!-- JAVASCRIPT LOGIC & CHARTS -->
  <!-- ======================================================================== -->
  <script>
    // 1. Toast function
    function showToast(msg) {
      const t = document.getElementById("toast");
      t.innerText = msg;
      t.className = "show";
      setTimeout(() => { t.className = t.className.replace("show", ""); }, 3000);
    }

    // 2. Copy card data handler
    function copyCardData(cardTitle, dataText) {
      navigator.clipboard.writeText(`[${cardTitle}] ${dataText}`).then(() => {
        showToast(`Copied ${cardTitle} data!`);
      });
    }

    // 3. Copy text summary
    function copyTextSummary() {
      const text = document.getElementById("rawTextContent").innerText;
      navigator.clipboard.writeText(text).then(() => {
        showToast("Copied text audit summary to clipboard!");
      });
    }

    // 4. Toggle Graphical / Text View
    function toggleTextView() {
      const gView = document.getElementById("graphicalView");
      const tView = document.getElementById("textView");
      const btn = document.getElementById("toggleViewBtn");

      if (tView.classList.contains("hidden")) {
        tView.classList.remove("hidden");
        gView.classList.add("hidden");
        btn.innerHTML = '<i class="fa-solid fa-chart-pie text-xs"></i> <span>Switch to Graphical View</span>';
      } else {
        tView.classList.add("hidden");
        gView.classList.remove("hidden");
        btn.innerHTML = '<i class="fa-solid fa-list-check text-xs"></i> <span>Switch to Text View</span>';
      }
    }

    // 5. Toggle Predictor Modal
    function togglePredictorModal() {
      const modal = document.getElementById("predictorModal");
      modal.classList.toggle("hidden");
    }

    // 6. Preset Scenario Loader
    function loadScenario(type) {
      if (type === 'ashtrang') {
        document.getElementById("inpCategory").value = "cultural";
        document.getElementById("inpCapacity").value = "500";
        document.getElementById("inpDuration").value = "180";
        document.getElementById("inpLeadDays").value = "21";
        document.getElementById("inpClashes").value = "0";
      } else if (type === 'hackathon') {
        document.getElementById("inpCategory").value = "technical";
        document.getElementById("inpCapacity").value = "250";
        document.getElementById("inpDuration").value = "480";
        document.getElementById("inpLeadDays").value = "14";
        document.getElementById("inpClashes").value = "0";
      } else if (type === 'clash') {
        document.getElementById("inpCategory").value = "fun_club";
        document.getElementById("inpCapacity").value = "100";
        document.getElementById("inpDuration").value = "90";
        document.getElementById("inpLeadDays").value = "1";
        document.getElementById("inpClashes").value = "3";
      }
    }

    // 7. Live Simulation API Caller
    async function runLiveSimulation(e) {
      e.preventDefault();
      const payload = {
        month: 11,
        day_of_week: 4,
        is_saturday: 0,
        week_of_year: 46,
        is_exam_season: 0,
        hour: 15,
        is_afternoon: 1,
        duration_min: parseInt(document.getElementById("inpDuration").value),
        lead_days: parseInt(document.getElementById("inpLeadDays").value),
        type_level: 2,
        designation_level: 2,
        room_no: 101,
        room_capacity: parseInt(document.getElementById("inpCapacity").value),
        faculty_event_count: 25,
        room_usage_count: 60,
        earlier_clash_count: parseInt(document.getElementById("inpClashes").value),
        cat_academic: document.getElementById("inpCategory").value === "academic" ? 1 : 0,
        cat_cultural: document.getElementById("inpCategory").value === "cultural" ? 1 : 0,
        cat_fun_club: document.getElementById("inpCategory").value === "fun_club" ? 1 : 0,
        cat_sports: document.getElementById("inpCategory").value === "sports" ? 1 : 0,
        cat_technical: document.getElementById("inpCategory").value === "technical" ? 1 : 0
      };

      try {
        const res = await fetch("/predict/all", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        
        document.getElementById("simResult").classList.remove("hidden");
        document.getElementById("resAttendance").innerText = `${data.attendance.predicted_registrations} students`;
        document.getElementById("resProb").innerText = `${(data.approval.probability_accept * 100).toFixed(1)}%`;
        document.getElementById("resBudget").innerText = `₹${Math.round(data.budget.predicted_budget_inr).toLocaleString()}`;
        
        const badge = document.getElementById("resBadge");
        if (data.approval.recommendation === "APPROVE") {
          badge.className = "px-2 py-0.5 rounded text-[11px] font-bold bg-emerald-100 text-emerald-800 border border-emerald-300";
          badge.innerText = "APPROVED";
        } else {
          badge.className = "px-2 py-0.5 rounded text-[11px] font-bold bg-rose-100 text-rose-800 border border-rose-300";
          badge.innerText = "CLASH / REJECT";
        }
      } catch (err) {
        showToast("Error running inference");
      }
    }

    // ========================================================================
    // INITIALIZE CHARTS MATCHING SCREENSHOT EXACTLY
    // ========================================================================
    window.addEventListener("DOMContentLoaded", () => {
      
      // Chart 1: Registrations Timeline Bar Chart (Teal & Charcoal alternating bars)
      const ctxBar = document.getElementById("registrationsBarChart").getContext("2d");
      new Chart(ctxBar, {
        type: "bar",
        data: {
          labels: [
            "", "", "January 28", "", "", "",
            "February 24", "", "", "", "",
            "May 19", "", "", "", "",
            "September 4", "", ""
          ],
          datasets: [{
            data: [
              4, 5, 12, 10, 11, 8, 12, 9, 11, 10, 12,
              25, 25, 9, 12, 8, 11, 12, 10
            ],
            backgroundColor: [
              "#2f3542", "#4ecdc4", "#2f3542", "#4ecdc4", "#2f3542", "#4ecdc4",
              "#2f3542", "#4ecdc4", "#2f3542", "#4ecdc4", "#2f3542",
              "#4ecdc4", "#2f3542", "#4ecdc4", "#2f3542", "#4ecdc4",
              "#2f3542", "#4ecdc4", "#2f3542"
            ],
            borderRadius: 2,
            barPercentage: 0.65
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { display: false },
            tooltip: {
              callbacks: {
                label: (ctx) => ` Registrations: ${ctx.parsed.y}`
              }
            }
          },
          scales: {
            x: {
              grid: { display: false },
              ticks: { font: { size: 9 }, color: "#64748b" }
            },
            y: {
              display: false,
              grid: { display: false }
            }
          }
        }
      });

      // Chart 2: Years Of Graduation Donut / Pie Chart (Teal, Dark Slate, Grey, Pink)
      const ctxDonut1 = document.getElementById("graduationDonutChart").getContext("2d");
      new Chart(ctxDonut1, {
        type: "pie",
        data: {
          labels: ["Unknown (General)", "2022", "2025", "2027"],
          datasets: [{
            data: [65, 15, 10, 10],
            backgroundColor: ["#4ecdc4", "#ced6e0", "#2f3542", "#ff4757"],
            borderWidth: 2,
            borderColor: "#ffffff"
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { display: false }
          }
        }
      });

      // Chart 3: Account Types Donut / Pie Chart
      const ctxDonut2 = document.getElementById("accountTypesDonutChart").getContext("2d");
      new Chart(ctxDonut2, {
        type: "pie",
        data: {
          labels: ["Undergraduate Student", "Parent/family", "Faculty/staff", "Graduate Student"],
          datasets: [{
            data: [55, 10, 15, 20],
            backgroundColor: ["#4ecdc4", "#ced6e0", "#2f3542", "#ff4757"],
            borderWidth: 2,
            borderColor: "#ffffff"
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { display: false }
          }
        }
      });

    });
  </script>
</body>
</html>
"""
