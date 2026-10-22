"""
Script to generate a professional ML System Architecture diagram for the College Event Management System.
Saves to ml_architecture.png.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_architecture():
    fig = plt.figure(figsize=(16, 11), dpi=300)
    fig.patch.set_facecolor('#0d1117')
    ax = fig.add_subplot(111)
    ax.set_facecolor('#0d1117')
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 11)
    ax.axis('off')

    # Styles
    title_color = '#58a6ff'
    text_color = '#e6edf3'
    subtext_color = '#8b949e'
    
    # Header
    ax.text(8, 10.5, "COLLEGE EVENT MANAGEMENT SYSTEM", ha='center', va='center',
            fontsize=20, fontweight='bold', color='#ffffff', family='sans-serif')
    ax.text(8, 10.12, "End-to-End MLOps & Triple-Task Machine Learning Architecture", ha='center', va='center',
            fontsize=12, fontweight='medium', color=title_color, family='sans-serif')

    def draw_box(x, y, w, h, title, subtitle, items, bg_color, border_col='#30363d', badge=None, badge_col='#238636'):
        rect = patches.FancyBboxPatch((x, y), w, h,
                                      boxstyle="round,pad=0.08,rounding_size=0.15",
                                      linewidth=1.5, edgecolor=border_col, facecolor=bg_color)
        ax.add_patch(rect)
        
        ax.text(x + w/2, y + h - 0.32, title, ha='center', va='center',
                fontsize=10.5, fontweight='bold', color='#ffffff', family='sans-serif')
        
        if subtitle:
            ax.text(x + w/2, y + h - 0.58, subtitle, ha='center', va='center',
                    fontsize=8.0, color=subtext_color, style='italic', family='sans-serif')
            start_y = y + h - 0.88
        else:
            start_y = y + h - 0.65
            
        for i, item in enumerate(items):
            ax.text(x + 0.22, start_y - (i * 0.28), item, ha='left', va='center',
                    fontsize=8.0, color=text_color, family='sans-serif')
            
        if badge:
            badge_rect = patches.FancyBboxPatch((x + w - 1.45, y + h - 0.32), 1.3, 0.23,
                                                boxstyle="round,pad=0.04,rounding_size=0.08",
                                                facecolor=badge_col, edgecolor='none')
            ax.add_patch(badge_rect)
            ax.text(x + w - 0.8, y + h - 0.20, badge, ha='center', va='center',
                    fontsize=7.0, fontweight='bold', color='#ffffff', family='sans-serif')

    def draw_arrow(x1, y1, x2, y2, label="", color="#58a6ff"):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->,head_width=0.35,head_length=0.45",
                                    color=color, lw=2, shrinkA=4, shrinkB=4))
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx, my + 0.12, label, ha='center', va='bottom',
                    fontsize=7.0, fontweight='bold', color='#79c0ff',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='#161b22', edgecolor='#30363d', alpha=0.9))

    # Stage Banners (y=9.45)
    ax.text(2.1, 9.45, "1. DATA & VERSIONING", ha='center', va='center',
            fontsize=10, fontweight='bold', color='#7ee787', family='sans-serif')
    ax.text(5.9, 9.45, "2. FEATURE EXTRACTION", ha='center', va='center',
            fontsize=10, fontweight='bold', color='#d29922', family='sans-serif')
    ax.text(10.0, 9.45, "3. MODELING PIPELINES", ha='center', va='center',
            fontsize=10, fontweight='bold', color='#a371f7', family='sans-serif')
    ax.text(13.9, 9.45, "4. MLOps & REGISTRY", ha='center', va='center',
            fontsize=10, fontweight='bold', color='#f778ba', family='sans-serif')

    # Column 1: Data Ingestion
    draw_box(0.6, 6.4, 3.0, 2.7, "RVCE 8th Mile Records", "rvce_8th_mile_data.csv",
             ["• 1,200 RVCE 8th Mile Records",
              "• Real Timetable Schedules",
              "• 16 Campus Venues (Main Stage)",
              "• Attendance (Registers)",
              "• Event Budget Allocation (₹)",
              "• Approval Status & Clash Engine"],
             bg_color='#161b22', border_col='#238636', badge="RVCE DATA", badge_col='#238636')
    
    draw_box(0.6, 3.2, 3.0, 2.8, "Data Lake Versioning", "lakeFS Architecture",
             ["• Branching: main / experiments",
              "• Zero-Copy Data Versioning",
              "• Immutable Commit Lineage",
              "• Object URIs on lakeFS",
              "• Reproducible snapshots"],
             bg_color='#161b22', border_col='#00d084', badge="LAKEFS TRACKED", badge_col='#00d084')

    # Column 2: Feature Engineering
    draw_box(4.2, 3.2, 3.4, 5.9, "Feature Extraction Engine", "eda_and_feature_extraction.py",
             ["• Spatio-Temporal Clash Engine",
              "   └ earlier_clash_count (Intervals)",
              "• Temporal & Calendar Signals",
              "   └ is_exam_season, month, lead_days",
              "• Hierarchical Encodings",
              "   └ type_level (Normal/Important/VIP)",
              "   └ designation_level (Asst/Assoc/Prof)",
              "• Entity Frequency Metrics",
              "   └ faculty_event_count, room_usage",
              "• Venue Physical Dynamics",
              "   └ room_capacity, fill_ratio",
              "• Genre Flags (5 Category One-Hots)",
              "─── Output: features_ready.csv ───"],
             bg_color='#161b22', border_col='#d29922', badge="22+ FEATURES", badge_col='#b08800')

    # Column 3: Triple ML Training Pipelines
    draw_box(8.2, 7.3, 3.6, 1.8, "Task A: Attendance Forecast", "TARGET_registers (Regression)",
             ["• Random Forest (Champion: R²=0.871)",
              "• Voting Regressor (R²=0.860)",
              "• Gradient Boosting (R²=0.850)",
              "• Turnout optimization & zero leakage"],
             bg_color='#1c152b', border_col='#8957e5', badge="CHAMPION: RF", badge_col='#8957e5')

    draw_box(8.2, 5.25, 3.6, 1.8, "Task B: Booking Approval", "TARGET_accept (Classification)",
             ["• Random Forest (Champion: AUC=0.828)",
              "• Voting Classifier (Accuracy=85.4%)",
              "• Gradient Boosting (Sensitivity=92.4%)",
              "• Automatic clash & overflow defense"],
             bg_color='#1c152b', border_col='#8957e5', badge="CHAMPION: RF", badge_col='#8957e5')

    draw_box(8.2, 3.2, 3.6, 1.8, "Task C: Budget Forecasting", "TARGET_budget (Regression)",
             ["• Gradient Boosting (Champion: R²=0.999)",
              "• Voting Regressor (R²=0.998)",
              "• Random Forest Regressor (R²=0.998)",
              "• Cost driver modeling (MAE: ₹3,325)"],
             bg_color='#1c152b', border_col='#2ea043', badge="CHAMPION: GBR", badge_col='#238636')

    # Column 4: MLOps, Tracking & Serving
    draw_box(12.4, 6.3, 3.0, 2.8, "Experiment Tracking", "Neptune.ai & MLflow",
             ["• Full Param & Metric Tracking",
              "• 5-Fold Stratified CV Runs",
              "• Residual Plots & Parity Curves",
              "• Artifact & Lineage Governance",
              "• Offline fallback + Cloud sync"],
             bg_color='#161b22', border_col='#0070f3', badge="MLFLOW & NEP", badge_col='#0070f3')

    draw_box(12.4, 3.2, 3.0, 2.8, "Model Registry & Metrics", "Production Gate & Reports",
             ["• @champion Production Tagging",
              "• college-attendance-champion",
              "• college-approval-champion",
              "• college-budget-champion",
              "• reports/champion_metrics.json"],
             bg_color='#161b22', border_col='#f778ba', badge="@CHAMPION", badge_col='#bf3989')

    # Bottom Stage: System Verification / CLI Layer
    draw_box(0.6, 0.7, 14.8, 2.1, "Automated Operational Workflow & Reproducibility Stack", "CLI & End-to-End Orchestration",
             ["• Step 1: `python3 eda_and_feature_extraction.py` ──> Extracts leak-free features and builds 22+ feature store",
              "• Step 2: `python3 train_budget_model.py` ──> Benchmarks 4 budget models, logs to MLflow/Neptune, serializes champion",
              "• Step 3: `python3 lakefs_versioning.py sync` ──> Commits raw records and feature versions with zero-copy isolation",
              "• Production Safeguards: 5-Fold CV, Class Imbalance Mitigation, Leakage Firewall across all 3 predictive tasks"],
             bg_color='#161b22', border_col='#388bfd')

    # Arrows between stages
    draw_arrow(3.6, 7.8, 4.2, 7.2, "Raw CSV")
    draw_arrow(2.1, 6.0, 2.1, 6.4, "Sync")
    draw_arrow(3.6, 4.6, 4.2, 4.6, "lakeFS Commit")

    # Feature Extraction to Pipelines
    draw_arrow(7.6, 7.5, 8.2, 8.2, "Attendance Features")
    draw_arrow(7.6, 6.1, 8.2, 6.1, "Approval Features")
    draw_arrow(7.6, 4.5, 8.2, 4.1, "Budget Features")

    # Pipelines to Neptune / Registry
    draw_arrow(11.8, 8.2, 12.4, 7.7, "Log Metrics")
    draw_arrow(11.8, 6.1, 12.4, 6.1, "Log Runs")
    draw_arrow(11.8, 4.1, 12.4, 4.5, "Register Model")

    # Registry Champion Tag
    draw_arrow(13.9, 6.3, 13.9, 6.0, "Champion Tag")

    # Pipeline to bottom layer
    draw_arrow(8.0, 3.2, 8.0, 2.8, "End-to-End Verification", color='#388bfd')

    plt.tight_layout()
    plt.savefig("ml_architecture.png", dpi=300, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close()
    print("Successfully generated ml_architecture.png with Task A, B, and C!")

if __name__ == "__main__":
    draw_architecture()
