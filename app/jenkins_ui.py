"""
Fake Jenkins CI/CD UI — matching real Jenkins 2.x design exactly.
Served via FastAPI at /jenkins and /cicd.
"""

JENKINS_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Jenkins</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Segoe UI', Arial, Helvetica, sans-serif; background: #f5f5f5; font-size: 13px; color: #333; }

    /* ══ TOP BAR ══ */
    #topbar {
      background: #3b8485;
      display: flex;
      align-items: center;
      height: 48px;
      padding: 0 16px;
      position: sticky;
      top: 0;
      z-index: 200;
    }
    .tb-logo {
      display: flex; align-items: center; gap: 8px;
      color: #fff; font-size: 22px; font-weight: 700;
      text-decoration: none; letter-spacing: 0.3px;
    }
    .tb-logo img { width: 36px; height: 36px; border-radius: 50%; background: #fff; }
    .jenkins-butler {
      width: 36px; height: 36px;
      background: #fff;
      border-radius: 50%;
      display: flex; align-items: center; justify-content: center;
      font-size: 22px;
      flex-shrink: 0;
    }
    .tb-search-wrap {
      flex: 1;
      display: flex;
      justify-content: center;
      padding: 0 24px;
    }
    .tb-search {
      display: flex; align-items: center; gap: 0;
      width: 100%; max-width: 440px;
    }
    .tb-search input {
      flex: 1;
      padding: 7px 12px;
      border: none;
      border-radius: 3px 0 0 3px;
      font-size: 13px;
      outline: none;
      background: #fff;
      color: #333;
    }
    .tb-search input::placeholder { color: #aaa; }
    .tb-search .search-btn {
      background: #fff;
      border: none;
      border-radius: 0 3px 3px 0;
      padding: 7px 10px;
      cursor: pointer;
      border-left: 1px solid #ddd;
      color: #666;
      font-size: 14px;
    }
    .tb-help {
      width: 28px; height: 28px;
      background: rgba(255,255,255,0.15);
      border: 1px solid rgba(255,255,255,0.3);
      border-radius: 50%;
      display: flex; align-items: center; justify-content: center;
      color: #fff; font-size: 14px; font-weight: bold;
      cursor: pointer; margin-left: 8px;
    }
    .tb-user {
      color: #fff; font-size: 12px; font-weight: 600;
      margin-left: 20px; display: flex; align-items: center; gap: 10px;
    }
    .tb-user a { color: #cef; text-decoration: none; font-size: 12px; }
    .tb-user a:hover { text-decoration: underline; }

    /* ══ BREADCRUMB BAR ══ */
    #breadbar {
      background: #fff;
      border-bottom: 1px solid #ddd;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 5px 16px;
      font-size: 12px;
    }
    .breadcrumb-left { display: flex; align-items: center; gap: 4px; color: #555; }
    .breadcrumb-left a { color: #296fa8; text-decoration: none; }
    .breadcrumb-left a:hover { text-decoration: underline; }
    .breadcrumb-left .sep { color: #aaa; }
    .breadcrumb-right a {
      font-size: 11px;
      color: #296fa8;
      text-decoration: none;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      font-weight: 600;
    }

    /* ══ LAYOUT ══ */
    .layout { display: flex; min-height: calc(100vh - 72px); }

    /* ══ SIDEBAR ══ */
    .sidebar {
      width: 220px;
      background: #fff;
      border-right: 1px solid #ddd;
      flex-shrink: 0;
      padding: 8px 0;
    }
    .sb-link {
      display: flex; align-items: center; gap: 10px;
      padding: 9px 16px;
      color: #333;
      font-size: 13px;
      cursor: pointer;
      text-decoration: none;
      transition: background 0.1s;
      border-left: 3px solid transparent;
    }
    .sb-link:hover { background: #f0f7f0; }
    .sb-link .sb-icon { font-size: 16px; width: 20px; text-align: center; color: #555; }
    .sb-link .new-icon {
      width: 18px; height: 18px;
      border: 2px solid #3b8485;
      border-radius: 50%;
      display: flex; align-items: center; justify-content: center;
      color: #3b8485;
      font-size: 14px;
      font-weight: bold;
      flex-shrink: 0;
    }

    .sb-section {
      margin-top: 12px;
      border-top: 1px solid #e8e8e8;
    }
    .sb-section-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 8px 14px 4px;
      font-size: 12px;
      font-weight: 700;
      color: #333;
      background: #f8f8f8;
      cursor: pointer;
      user-select: none;
    }
    .sb-section-header .arrow { font-size: 10px; color: #777; }
    .sb-section-body { padding: 6px 14px 10px; font-size: 12px; color: #666; }

    /* ══ MAIN CONTENT ══ */
    .main { flex: 1; padding: 0; overflow-y: auto; }

    /* ══ TABS ══ */
    .tab-bar {
      display: flex;
      align-items: center;
      background: #fff;
      border-bottom: 1px solid #ddd;
      padding: 0 16px;
      position: relative;
    }
    .tab-item {
      padding: 10px 18px 8px;
      cursor: pointer;
      font-size: 13px;
      font-weight: 600;
      color: #333;
      border-bottom: 2px solid transparent;
      margin-bottom: -1px;
    }
    .tab-item.active {
      color: #3b8485;
      border-bottom-color: #3b8485;
    }
    .tab-add {
      width: 26px; height: 26px;
      border: 1px solid #bbb;
      border-radius: 3px;
      display: flex; align-items: center; justify-content: center;
      cursor: pointer;
      color: #555; font-size: 16px;
      margin-left: 6px;
    }
    .tab-add:hover { background: #f0f0f0; }
    .tab-bar-right {
      margin-left: auto;
      display: flex; align-items: center; gap: 8px;
    }
    .pencil-link { color: #296fa8; font-size: 12px; text-decoration: none; display: flex; align-items: center; gap: 4px; }
    .pencil-link:hover { text-decoration: underline; }

    /* ══ JOBS TABLE ══ */
    .table-wrap { padding: 0; }
    table { width: 100%; border-collapse: collapse; background: #fff; }
    thead th {
      background: #fff;
      border-top: none;
      border-bottom: 2px solid #ddd;
      padding: 9px 10px;
      text-align: left;
      font-size: 12px;
      font-weight: 700;
      color: #333;
      cursor: pointer;
      white-space: nowrap;
    }
    thead th:hover { background: #f5f5f5; }
    thead th .sort-arrow { color: #3b8485; margin-left: 3px; font-size: 10px; }
    td {
      padding: 9px 10px;
      border-bottom: 1px solid #eee;
      vertical-align: middle;
      font-size: 13px;
    }
    tr:hover td { background: #f7fdf7; }
    tr:last-child td { border-bottom: none; }

    /* Status icon */
    .status-icon {
      width: 28px; height: 28px;
      border-radius: 50%;
      display: flex; align-items: center; justify-content: center;
      font-size: 16px;
    }
    .status-ok { background: #3b8485; color: #fff; }
    .status-fail { background: #c0392b; color: #fff; }
    .status-unstable { background: #e8a838; color: #fff; }

    /* Weather icon */
    .weather-icon { font-size: 22px; }

    /* Job name link */
    .job-name-cell { display: flex; align-items: center; gap: 0; }
    .job-name {
      color: #296fa8; font-size: 13px; font-weight: 600;
      text-decoration: none; cursor: pointer;
    }
    .job-name:hover { text-decoration: underline; }
    .job-dropdown {
      color: #666; cursor: pointer; font-size: 10px; margin-left: 4px;
      padding: 2px 4px;
    }
    .job-dropdown:hover { background: #eee; border-radius: 2px; }

    /* Success/failure links */
    .time-link { color: #296fa8; text-decoration: none; }
    .time-link:hover { text-decoration: underline; }

    /* Play button */
    .play-btn {
      width: 28px; height: 28px;
      border: 1px solid #bbb;
      border-radius: 50%;
      display: flex; align-items: center; justify-content: center;
      cursor: pointer;
      color: #555;
      font-size: 12px;
      transition: all 0.15s;
      flex-shrink: 0;
    }
    .play-btn:hover { background: #3b8485; color: #fff; border-color: #3b8485; }

    /* ══ ICON SIZE SELECTOR ══ */
    .icon-size-row {
      padding: 4px 10px 8px;
      font-size: 12px;
      color: #555;
      display: flex;
      align-items: center;
      gap: 4px;
    }
    .icon-size-row .label { margin-right: 4px; }
    .icon-size-row a { color: #296fa8; text-decoration: none; cursor: pointer; }
    .icon-size-row a:hover { text-decoration: underline; }

    /* ══ RSS ROW ══ */
    .rss-row {
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 10px 10px 14px;
      font-size: 12px;
      color: #333;
      flex-wrap: wrap;
    }
    .rss-link {
      display: flex; align-items: center; gap: 4px;
      color: #c47c2a;
      text-decoration: none;
      font-weight: 600;
      font-size: 12px;
    }
    .rss-icon { font-size: 14px; color: #e68029; }

    /* ══ CONTEXT MENU ══ */
    #ctxMenu {
      position: fixed;
      background: #fff;
      border: 1px solid #bbb;
      border-radius: 3px;
      box-shadow: 2px 4px 12px rgba(0,0,0,0.18);
      min-width: 210px;
      z-index: 999;
      display: none;
      overflow: hidden;
    }
    .ctx-item {
      display: flex; align-items: center; gap: 10px;
      padding: 10px 16px;
      cursor: pointer;
      font-size: 13px;
      color: #333;
      border-bottom: 1px solid #f0f0f0;
      transition: background 0.1s;
    }
    .ctx-item:last-child { border-bottom: none; }
    .ctx-item:hover { background: #f0f7f0; }
    .ctx-item.active { background: #3b8485; color: #fff; }
    .ctx-item.active .ctx-icon { color: #fff; }
    .ctx-icon { font-size: 15px; width: 18px; text-align: center; color: #555; }

    /* ══ BUILD QUEUE / EXECUTOR ══ */
    .sb-executor {
      display: flex; align-items: center; gap: 8px;
      padding: 5px 14px;
      font-size: 12px; color: #555;
    }
    .executor-num { font-weight: 700; color: #333; }

    /* ══ Toast ══ */
    #jk-toast {
      position: fixed; bottom: 24px; right: 24px;
      background: #333; color: #fff;
      padding: 10px 18px; border-radius: 3px;
      font-size: 12px; z-index: 9999;
      display: none;
      box-shadow: 0 4px 16px rgba(0,0,0,0.25);
    }
    .build-running-row td { background: #fffde7 !important; }
  </style>
</head>
<body>

<!-- ══ TOP BAR ══ -->
<div id="topbar">
  <a class="tb-logo" href="#">
    <div class="jenkins-butler">🤵</div>
    <span>Jenkins</span>
  </a>
  <div class="tb-search-wrap">
    <div class="tb-search">
      <input type="text" placeholder="search" id="searchBox">
      <button class="search-btn">🔍</button>
    </div>
  </div>
  <div class="tb-help">?</div>
  <div class="tb-user">
    AARAV J
    <a href="#" onclick="jkToast('Logging out...')">| LOG OUT</a>
  </div>
</div>

<!-- ══ BREADCRUMB BAR ══ -->
<div id="breadbar">
  <div class="breadcrumb-left">
    <a href="#">Jenkins</a>
    <span class="sep">›</span>
  </div>
  <div class="breadcrumb-right">
    <a href="#" onclick="toggleAutoRefresh(this)">ENABLE AUTO REFRESH</a>
  </div>
</div>

<!-- ══ LAYOUT ══ -->
<div class="layout">

  <!-- SIDEBAR -->
  <aside class="sidebar">
    <a class="sb-link" href="#" onclick="jkToast('Opening New Item wizard...')">
      <div class="new-icon">+</div> New Item
    </a>
    <a class="sb-link" href="#"><span class="sb-icon">👥</span> People</a>
    <a class="sb-link" href="#" onclick="switchToHistory()"><span class="sb-icon">🕐</span> Build History</a>
    <a class="sb-link" href="#"><span class="sb-icon">⚙️</span> Manage Jenkins</a>
    <a class="sb-link" href="#"><span class="sb-icon">🔑</span> Credentials</a>
    <a class="sb-link" href="#"><span class="sb-icon">👁️</span> My Views</a>

    <div class="sb-section">
      <div class="sb-section-header" onclick="toggleSection('buildqueue')">
        Build Queue <span class="arrow" id="bq-arrow">▲</span>
      </div>
      <div class="sb-section-body" id="buildqueue">
        No builds in the queue.
      </div>
    </div>

    <div class="sb-section">
      <div class="sb-section-header" onclick="toggleSection('executor')">
        Build Executor Status <span class="arrow" id="be-arrow">▲</span>
      </div>
      <div class="sb-section-body" id="executor" style="padding:4px 0 8px;">
        <div class="sb-executor"><span class="executor-num">1</span> Idle</div>
        <div class="sb-executor"><span class="executor-num">2</span> <span id="exec2-status">Idle</span></div>
      </div>
    </div>
  </aside>

  <!-- MAIN -->
  <main class="main">
    <div id="view-dashboard">
      <!-- TAB BAR -->
      <div class="tab-bar">
        <div class="tab-item active" id="tab-all">All</div>
        <div class="tab-add">+</div>
        <div class="tab-bar-right">
          <a class="pencil-link" href="#" onclick="jkToast('Opening description editor...')">✏️ add description</a>
        </div>
      </div>

      <!-- JOBS TABLE -->
      <div class="table-wrap">
        <table id="jobsTable">
          <thead>
            <tr>
              <th style="width:36px;">S <span class="sort-arrow">↑</span></th>
              <th style="width:36px;">W</th>
              <th>Name</th>
              <th>Last Success</th>
              <th>Last Failure</th>
              <th>Last Duration</th>
              <th style="width:36px;"></th>
            </tr>
          </thead>
          <tbody id="jobsTbody">
            <!-- Row 1: MLOps Pipeline -->
            <tr id="row-mlops">
              <td><div class="status-icon status-ok" title="Last build succeeded">✔</div></td>
              <td><div class="weather-icon" title="Build stable">☀️</div></td>
              <td>
                <div class="job-name-cell">
                  <a class="job-name" href="#" onclick="openJob('mlops')">college-event-management-mlops</a>
                  <span class="job-dropdown" onclick="showCtxMenu(event, 'mlops')">▼</span>
                </div>
              </td>
              <td><a class="time-link" href="#" onclick="openBuild(9)">3 hr - #9</a></td>
              <td><a class="time-link" href="#" onclick="openBuild(7)">5 hr - #7</a></td>
              <td>4 min 12 sec</td>
              <td><div class="play-btn" onclick="triggerBuild('mlops')" title="Build Now">▶</div></td>
            </tr>
            <!-- Row 2: Vercel Deploy -->
            <tr id="row-vercel">
              <td><div class="status-icon status-ok" title="Last build succeeded">✔</div></td>
              <td><div class="weather-icon" title="Build stable">🌤️</div></td>
              <td>
                <div class="job-name-cell">
                  <a class="job-name" href="#" onclick="openJob('vercel')">college-event-management-vercel-deploy</a>
                  <span class="job-dropdown" onclick="showCtxMenu(event, 'vercel')">▼</span>
                </div>
              </td>
              <td><a class="time-link" href="#" onclick="openBuild(9)">3 hr - #9</a></td>
              <td><a class="time-link" href="#" onclick="openBuild(7)">1 day 2 hr - #7</a></td>
              <td>2 min 7 sec</td>
              <td><div class="play-btn" onclick="triggerBuild('vercel')" title="Build Now">▶</div></td>
            </tr>
            <!-- Row 3: Docker Build -->
            <tr id="row-docker">
              <td><div class="status-icon status-ok" title="Last build succeeded">✔</div></td>
              <td><div class="weather-icon" title="Build stable">☀️</div></td>
              <td>
                <div class="job-name-cell">
                  <a class="job-name" href="#" onclick="openJob('docker')">college-event-intelligence-docker</a>
                  <span class="job-dropdown" onclick="showCtxMenu(event, 'docker')">▼</span>
                </div>
              </td>
              <td><a class="time-link" href="#" onclick="openBuild(4)">3 hr - #4</a></td>
              <td>N/A</td>
              <td>1 min 15 sec</td>
              <td><div class="play-btn" onclick="triggerBuild('docker')" title="Build Now">▶</div></td>
            </tr>
            <!-- Row 4: Model Quality Gate -->
            <tr id="row-quality">
              <td><div class="status-icon status-ok" title="Last build succeeded">✔</div></td>
              <td><div class="weather-icon" title="Build stable">☀️</div></td>
              <td>
                <div class="job-name-cell">
                  <a class="job-name" href="#" onclick="openJob('quality')">model-quality-gates-45-tests</a>
                  <span class="job-dropdown" onclick="showCtxMenu(event, 'quality')">▼</span>
                </div>
              </td>
              <td><a class="time-link" href="#" onclick="openBuild(9)">3 hr - #9</a></td>
              <td><a class="time-link" href="#" onclick="openBuild(3)">22 hr - #3</a></td>
              <td>44 sec</td>
              <td><div class="play-btn" onclick="triggerBuild('quality')" title="Build Now">▶</div></td>
            </tr>
            <!-- Row 5: TensorBoard Tracking -->
            <tr id="row-tb">
              <td><div class="status-icon status-unstable" title="Unstable">!</div></td>
              <td><div class="weather-icon" title="Some failures">⛅</div></td>
              <td>
                <div class="job-name-cell">
                  <a class="job-name" href="#" onclick="openJob('tb')">tensorboard-experiment-sync</a>
                  <span class="job-dropdown" onclick="showCtxMenu(event, 'tb')">▼</span>
                </div>
              </td>
              <td><a class="time-link" href="#">1 day - #2</a></td>
              <td><a class="time-link" href="#">6 hr - #3</a></td>
              <td>18 sec</td>
              <td><div class="play-btn" onclick="triggerBuild('tb')" title="Build Now">▶</div></td>
            </tr>
          </tbody>
        </table>

        <!-- Icon size row -->
        <div class="icon-size-row">
          <span class="label">Icon:</span>
          <a href="#" onclick="jkToast('Icon size: S')">S</a>&nbsp;
          <a href="#" onclick="jkToast('Icon size: M')">M</a>&nbsp;
          <a href="#" onclick="jkToast('Icon size: L')">L</a>
        </div>

        <!-- RSS row -->
        <div class="rss-row">
          <a class="rss-link" href="#"><span class="rss-icon">📡</span> RSS for all</a>
          <a class="rss-link" href="#"><span class="rss-icon">📡</span> RSS for failures</a>
          <a class="rss-link" href="#"><span class="rss-icon">📡</span> RSS for just latest builds</a>
        </div>
      </div>
    </div>

    <!-- BUILD HISTORY VIEW -->
    <div id="view-history" style="display:none; padding:16px;">
      <h2 style="margin-bottom:14px;font-size:16px;color:#333;">Build History</h2>
      <table>
        <thead>
          <tr>
            <th>#</th><th>Job</th><th>Status</th><th>Branch</th><th>Commit</th><th>Started</th><th>Duration</th>
          </tr>
        </thead>
        <tbody>
          <tr><td><a class="time-link" href="#" onclick="backToDash()">#9</a></td><td>college-event-management-mlops</td><td>✅ Success</td><td>main</td><td><code>da8f718</code></td><td>09:49 IST</td><td>4m 12s</td></tr>
          <tr><td><a class="time-link" href="#">#8</a></td><td>college-event-management-mlops</td><td>✅ Success</td><td>main</td><td><code>05694e3</code></td><td>09:39 IST</td><td>5m 08s</td></tr>
          <tr><td><a class="time-link" href="#">#7</a></td><td>college-event-management-mlops</td><td>❌ Failed</td><td>main</td><td><code>87b692e</code></td><td>09:19 IST</td><td>2m 33s</td></tr>
          <tr><td><a class="time-link" href="#">#6</a></td><td>college-event-management-mlops</td><td>✅ Success</td><td>main</td><td><code>3b8948d</code></td><td>08:52 IST</td><td>4m 55s</td></tr>
          <tr><td><a class="time-link" href="#">#5</a></td><td>college-event-management-mlops</td><td>✅ Success</td><td>main</td><td><code>45f2fc8</code></td><td>08:21 IST</td><td>6m 01s</td></tr>
          <tr><td><a class="time-link" href="#">#4</a></td><td>college-event-intelligence-docker</td><td>✅ Success</td><td>main</td><td><code>870d9e0</code></td><td>07:44 IST</td><td>1m 15s</td></tr>
          <tr><td><a class="time-link" href="#">#3</a></td><td>model-quality-gates-45-tests</td><td>⚠️ Unstable</td><td>main</td><td><code>dbfa9bf</code></td><td>07:12 IST</td><td>44s</td></tr>
          <tr><td><a class="time-link" href="#">#2</a></td><td>college-event-management-mlops</td><td>✅ Success</td><td>main</td><td><code>633437c</code></td><td>Yesterday</td><td>4m 38s</td></tr>
          <tr><td><a class="time-link" href="#">#1</a></td><td>college-event-management-mlops</td><td>✅ Success</td><td>main</td><td><code>84273e4</code></td><td>Nov 10</td><td>8m 11s</td></tr>
        </tbody>
      </table>
      <div style="margin-top:12px;"><a class="time-link" href="#" onclick="backToDash()">← Back to Dashboard</a></div>
    </div>
  </main>
</div>

<!-- ══ CONTEXT MENU ══ -->
<div id="ctxMenu">
  <div class="ctx-item" onclick="ctxAction('Changes')"><span class="ctx-icon">🕐</span> Changes</div>
  <div class="ctx-item" onclick="ctxAction('Workspace')"><span class="ctx-icon">📁</span> Workspace</div>
  <div class="ctx-item" onclick="ctxBuildNow()"><span class="ctx-icon">▶</span> Build Now</div>
  <div class="ctx-item" onclick="ctxAction('DeleteProject')"><span class="ctx-icon">✖</span> Delete Project</div>
  <div class="ctx-item" onclick="ctxAction('Configure')"><span class="ctx-icon">⚙️</span> Configure</div>
  <div class="ctx-item active" id="ctxGitHub"><span class="ctx-icon">🔗</span> GitHub Hook Log</div>
  <div class="ctx-item" onclick="ctxAction('Move')"><span class="ctx-icon">➡️</span> Move</div>
</div>

<!-- ══ TOAST ══ -->
<div id="jk-toast"></div>

<script>
  let ctxTarget = null;
  let autoRefresh = false;
  let autoRefreshTimer = null;

  // ── Context menu ──
  function showCtxMenu(e, job) {
    e.preventDefault(); e.stopPropagation();
    ctxTarget = job;
    const menu = document.getElementById('ctxMenu');
    menu.style.display = 'block';
    // Position
    let x = e.clientX, y = e.clientY;
    if (x + 220 > window.innerWidth) x = window.innerWidth - 230;
    if (y + 260 > window.innerHeight) y = window.innerHeight - 270;
    menu.style.left = x + 'px';
    menu.style.top = y + 'px';
  }
  document.addEventListener('click', () => {
    document.getElementById('ctxMenu').style.display = 'none';
  });
  function ctxAction(name) {
    document.getElementById('ctxMenu').style.display = 'none';
    jkToast('Opening ' + name + '...');
  }
  function ctxBuildNow() {
    document.getElementById('ctxMenu').style.display = 'none';
    if (ctxTarget) triggerBuild(ctxTarget);
  }

  // ── Trigger build ──
  function triggerBuild(job) {
    const rowId = 'row-' + job;
    const row = document.getElementById(rowId);
    if (!row) { jkToast('Build queued for: ' + job); return; }

    // Flash the row
    row.classList.add('build-running-row');
    const statusCell = row.querySelector('.status-icon');
    statusCell.style.background = '#1a6fa6';
    statusCell.textContent = '⟳';
    statusCell.title = 'Building...';

    // Update executor
    document.getElementById('exec2-status').textContent = 'Building ' + job;

    jkToast('Build queued — ' + job);

    setTimeout(() => {
      statusCell.style.background = '#3b8485';
      statusCell.textContent = '✔';
      statusCell.title = 'Last build succeeded';
      row.classList.remove('build-running-row');
      // Update "Last Success"
      const cells = row.querySelectorAll('td');
      const buildNum = Math.floor(Math.random() * 3) + 10;
      cells[3].innerHTML = `<a class="time-link" href="#">just now - #${buildNum}</a>`;
      cells[5].textContent = Math.floor(Math.random() * 4 + 1) + ' min ' + Math.floor(Math.random() * 59 + 1) + ' sec';
      document.getElementById('exec2-status').textContent = 'Idle';
      jkToast('✅ Build #' + buildNum + ' succeeded for ' + job + '!');
    }, 3500);
  }

  function openBuild(n) {
    jkToast('Opening build #' + n + ' console output...');
  }
  function openJob(j) {
    jkToast('Opening pipeline view for: ' + j + '...');
  }

  // ── View switching ──
  function switchToHistory() {
    document.getElementById('view-dashboard').style.display = 'none';
    document.getElementById('view-history').style.display = 'block';
  }
  function backToDash() {
    document.getElementById('view-history').style.display = 'none';
    document.getElementById('view-dashboard').style.display = 'block';
  }

  // ── Sections collapse ──
  function toggleSection(id) {
    const el = document.getElementById(id);
    const arrow = document.getElementById(id === 'buildqueue' ? 'bq-arrow' : 'be-arrow');
    if (el.style.display === 'none') {
      el.style.display = '';
      arrow.textContent = '▲';
    } else {
      el.style.display = 'none';
      arrow.textContent = '▼';
    }
  }

  // ── Auto refresh ──
  function toggleAutoRefresh(el) {
    autoRefresh = !autoRefresh;
    if (autoRefresh) {
      el.textContent = 'DISABLE AUTO REFRESH';
      autoRefreshTimer = setInterval(() => {
        jkToast('🔄 Auto-refreshed');
      }, 15000);
    } else {
      el.textContent = 'ENABLE AUTO REFRESH';
      clearInterval(autoRefreshTimer);
    }
  }

  // ── Search filter ──
  document.getElementById('searchBox').addEventListener('input', function() {
    const q = this.value.toLowerCase();
    document.querySelectorAll('#jobsTbody tr').forEach(row => {
      const name = row.querySelector('.job-name');
      if (!name) return;
      row.style.display = name.textContent.toLowerCase().includes(q) ? '' : 'none';
    });
  });

  // ── Toast ──
  function jkToast(msg) {
    const t = document.getElementById('jk-toast');
    t.textContent = msg;
    t.style.display = 'block';
    clearTimeout(jkToast._t);
    jkToast._t = setTimeout(() => { t.style.display = 'none'; }, 2500);
  }
</script>
</body>
</html>
"""
