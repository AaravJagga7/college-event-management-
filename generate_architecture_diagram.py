"""
Script to generate a professional ML System Architecture diagram for the College Event Management System.
Saves to ml_architecture.png.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_architecture():
    fig = plt.figure(figsize=(16, 10), dpi=300)
    fig.patch.set_facecolor('#0d1117')
    ax = fig.add_subplot(111)
    ax.set_facecolor('#0d1117')
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Styles
    title_color = '#58a6ff'
    text_color = '#e6edf3'
    subtext_color = '#8b949e'
    border_color = '#30363d'
    
    # Header
    ax.text(8, 9.5, "COLLEGE EVENT MANAGEMENT SYSTEM", ha='center', va='center',
            fontsize=20, fontweight='bold', color='#ffffff', family='sans-serif')
    ax.text(8, 9.15, "End-to-End MLOps & Dual-Task Machine Learning Architecture", ha='center', va='center',
            fontsize=12, fontweight='medium', color=title_color, family='sans-serif')

    def draw_box(x, y, w, h, title, subtitle, items, bg_color, border_col='#30363d', badge=None, badge_col='#238636'):
        # Rounded rectangle
        rect = patches.FancyBboxPatch((x, y), w, h,
                                      boxstyle="round,pad=0.08,rounding_size=0.15",
                                      linewidth=1.5, edgecolor=border_col, facecolor=bg_color)
        ax.add_patch(rect)
        
        # Title
        ax.text(x + w/2, y + h - 0.35, title, ha='center', va='center',
                fontsize=11, fontweight='bold', color='#ffffff', family='sans-serif')
        
        if subtitle:
            ax.text(x + w/2, y + h - 0.62, subtitle, ha='center', va='center',
                    fontsize=8.5, color=subtext_color, style='italic', family='sans-serif')
            start_y = y + h - 0.95
        else:
            start_y = y + h - 0.70
            
        # Items
        for i, item in enumerate(items):
            ax.text(x + 0.25, start_y - (i * 0.32), item, ha='left', va='center',
                    fontsize=8.5, color=text_color, family='sans-serif')
            
        if badge:
            badge_rect = patches.FancyBboxPatch((x + w - 1.45, y + h - 0.35), 1.3, 0.25,
                                                boxstyle="round,pad=0.04,rounding_size=0.08",
                                                facecolor=badge_col, edgecolor='none')
            ax.add_patch(badge_rect)
            ax.text(x + w - 0.8, y + h - 0.22, badge, ha='center', va='center',
                    fontsize=7.5, fontweight='bold', color='#ffffff', family='sans-serif')

    def draw_arrow(x1, y1, x2, y2, label="", color="#58a6ff"):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->,head_width=0.35,head_length=0.45",
                                    color=color, lw=2, shrinkA=4, shrinkB=4))
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx, my + 0.15, label, ha='center', va='bottom',
                    fontsize=7.5, fontweight='bold', color='#79c0ff',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='#161b22', edgecolor='#30363d', alpha=0.9))

    # Column 1: Data Ingestion (x=0.6, w=3.0)
    # Stage Banner
    ax.text(2.1, 8.5, "1. DATA & VERSIONING", ha='center', va='center',
            fontsize=10, fontweight='bold', color='#7ee787', family='sans-serif')
    
    draw_box(0.6, 6.2, 3.0, 2.0, "Raw Event Records", "synthetic_event_data.csv",
             ["• 1,200 Historical Event Bookings",
              "• Faculty & Department Metadata",
              "• Room Capacities & Time Slots",
              "• Booking Status & Reg. Attendance"],
             bg_color='#161b22', border_col='#238636', badge="DATA V1", badge_col='#238636')
    
    draw_box(0.6, 3.7, 3.0, 2.0, "Data Version Control", "DVC Pipeline Management",
             ["• dvc.yaml & dvc.lock Tracking",
              "• Hash-based Data Checksums",
              "• Immutable Pipeline Stages",
              "• Deterministic Repro (`dvc repro`)"],
             bg_color='#161b22', border_col='#1f6feb', badge="DVC TRACKED", badge_col='#1f6feb')

    # Column 2: Feature Engineering (x=4.2, w=3.4)
    ax.text(5.9, 8.5, "2. FEATURE EXTRACTION", ha='center', va='center',
            fontsize=10, fontweight='bold', color='#d29922', family='sans-serif')
    
    draw_box(4.2, 3.7, 3.4, 4.5, "Feature Extraction Engine", "eda_and_feature_extraction.py",
             ["• Spatio-Temporal Clash Engine",
              "   └ earlier_clash_count (Interval Overlaps)",
              "• Temporal & Calendar Signals",
              "   └ is_exam_season, month, hour, lead_days",
              "• Hierarchical Encodings",
              "   └ type_level (Normal/Important/VIP)",
              "   └ designation_level (Asst/Assoc/Prof)",
              "• Entity Frequency Metrics",
              "   └ faculty_event_count, room_usage_count",
              "• Room Physical Dynamics",
              "   └ room_capacity, fill_ratio interaction",
              "• Categorical One-Hot Flags (5 classes)",
              "─── Output: features_ready.csv ───"],
             bg_color='#161b22', border_col='#d29922', badge="21 FEATURES", badge_col='#b08800')

    # Column 3: Dual ML Training Pipelines (x=8.2, w=3.6)
    ax.text(10.0, 8.5, "3. MODELING PIPELINES", ha='center', va='center',
            fontsize=10, fontweight='bold', color='#a371f7', family='sans-serif')
    
    draw_box(8.2, 6.1, 3.6, 2.1, "Task A: Attendance Forecast", "TARGET_registers (Regression)",
             ["• Gradient Boosting (Champion: R²=0.558)",
              "• Voting Regressor (R²=0.526)",
              "• Random Forest Regressor (R²=0.505)",
              "• Decision Tree Regressor (R²=0.366)",
              "• Prevents venue shortage & misallocation"],
             bg_color='#1c152b', border_col='#8957e5', badge="CHAMPION: GBR", badge_col='#8957e5')

    draw_box(8.2, 3.7, 3.6, 2.1, "Task B: Booking Approval", "TARGET_accept (Classification)",
             ["• Random Forest (Champion: AUC=0.920)",
              "• Voting Classifier (Soft Ensemble)",
              "• Gradient Boosting (Sensitivity=99.1%)",
              "• Decision Tree (Baseline Benchmark)",
              "• Prevents room clash & capacity violations"],
             bg_color='#1c152b', border_col='#8957e5', badge="CHAMPION: RF", badge_col='#8957e5')

    # Column 4: MLOps, Tracking & Serving (x=12.4, w=3.0)
    ax.text(13.9, 8.5, "4. MLOps & REGISTRY", ha='center', va='center',
            fontsize=10, fontweight='bold', color='#f778ba', family='sans-serif')
    
    draw_box(12.4, 6.1, 3.0, 2.1, "MLflow Experiment Tracking", "sqlite:///mlflow.db & Artifacts",
             ["• Full Hyperparameter Logging",
              "• 5-Fold Stratified CV Scores",
              "• Confusion Matrices & ROC Curves",
              "• Model Signatures & Input Examples"],
             bg_color='#161b22', border_col='#da3633', badge="MLFLOW UI", badge_col='#da3633')

    draw_box(12.4, 3.7, 3.0, 2.1, "Model Registry & Metrics", "Production Deployment Gate",
             ["• @champion Production Alias",
              "• college-approval-champion",
              "• reports/champion_metrics.json",
              "• Automated Benchmarking Reports"],
             bg_color='#161b22', border_col='#f778ba', badge="@CHAMPION", badge_col='#bf3989')

    # Bottom Stage: System Verification / CLI Layer (x=0.6, y=0.8, w=14.8, h=2.2)
    draw_box(0.6, 0.8, 14.8, 2.2, "Automated Operational Workflow & Reproducibility Stack", "CLI & End-to-End Orchestration",
             ["• Step 1: `dvc repro` ──────> Executes deterministic DAG: Feature Extraction -> Model Training -> Metrics Evaluation",
              "• Step 2: `python3 calculate_scores.py` ──> Generates champion_metrics.json & detailed multi-metric CSV reports",
              "• Step 3: `mlflow ui --backend-store-uri sqlite:///mlflow.db` ──> Serves interactive visualization dashboard on localhost:5000",
              "• Production Safeguards: Stratified 5-Fold CV, Class Imbalance Handling (class_weight='balanced'), Zero Target Leakage"],
             bg_color='#161b22', border_col='#388bfd')

    # Arrows between stages
    # Data to Feature Extraction
    draw_arrow(3.6, 7.2, 4.2, 6.5, "Raw CSV")
    draw_arrow(2.1, 5.7, 2.1, 6.2, "Sync")
    draw_arrow(3.6, 4.7, 4.2, 5.0, "DVC DAG")

    # Feature Extraction to Pipelines
    draw_arrow(7.6, 6.3, 8.2, 7.15, "Features (No Leak)")
    draw_arrow(7.6, 4.7, 8.2, 4.75, "All Features")

    # Pipelines to MLflow
    draw_arrow(11.8, 7.15, 12.4, 7.15, "Log Metrics")
    draw_arrow(11.8, 4.75, 12.4, 4.75, "Register Model")

    # MLflow to Registry
    draw_arrow(13.9, 6.1, 13.9, 5.8, "Alias Tagging")

    # Pipeline to bottom layer
    draw_arrow(8.0, 3.7, 8.0, 3.0, "End-to-End Validation", color='#388bfd')

    plt.tight_layout()
    plt.savefig("ml_architecture.png", dpi=300, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close()
    print("Successfully generated ml_architecture.png!")

if __name__ == "__main__":
    draw_architecture()
