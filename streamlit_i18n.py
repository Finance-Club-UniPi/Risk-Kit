"""UI strings for Crisis Radar Streamlit app (EN / EL for Finance Club members)."""

from __future__ import annotations

STRINGS: dict[str, dict[str, str]] = {
    "en": {
        "lang_label": "Language",
        "nav_header": "Navigation",
        "page_home": "Home",
        "page_train": "Train Model",
        "page_live": "Live Predictions",
        "page_test": "Test Data",
        "page_viz": "Visualizations",
        "page_about": "About",
        "sidebar_model_ready": "Model trained and ready",
        "sidebar_no_model": "No trained model found. Train a model first.",
        "sidebar_credits": "**Developed by:** Apostolos Chardalias | Finance Club UniPi",
        "sidebar_version": "**Version:** 1.1.0",
        "home_title": "Welcome to Crisis Radar",
        "home_tagline": "Use calibrated probabilities to <strong>inform</strong> your view on drawdown risk — not to replace judgment.",
        "home_intro": (
            "**Crisis Radar** estimates the probability of a large S&P 500 drawdown over the next *N* trading days, "
            "using history, volatility, and VIX-style inputs. It is built for learning and research."
        ),
        "home_features_title": "What you can do",
        "home_features": (
            "- **Train** Random Forest, Gradient Boosting, or Logistic Regression with a strict time split\n"
            "- **Live score** for the latest session using yfinance data\n"
            "- **Upload CSV** to score your own aligned GSPC + VIX series\n"
            "- **Charts** for calibration, PR curve, confusion matrix, feature importance"
        ),
        "home_qs_title": "Quick path (first visit)",
        "home_qs": (
            "1. **Train Model** — use defaults or adjust horizon/threshold, then train (may take a few minutes).\n"
            "2. **Live Predictions** — refresh if you need the very latest prices.\n"
            "3. **Test Data** — download the sample CSV, replace with your slice, upload.\n"
            "4. **Visualizations** — check calibration and whether the model fits your use case."
        ),
        "home_decision_title": "How to turn output into better decisions",
        "home_decision": (
            "- **Risk bands**: LOW &lt;10%, MEDIUM 10–30%, HIGH ≥30% (default cutoffs; not trading signals).\n"
            "- **Trust calibration**: if the calibration curve is poor, treat probabilities as rough ranks, not exact %%.\n"
            "- **PR-AUC / F1**: useful on imbalanced crisis labels; compare to a naive baseline before acting.\n"
            "- **Horizon & threshold**: a 20-day −8%% event is not the same as “market crash tomorrow.”\n"
            "- **Disclaimer**: educational/research only — not financial advice."
        ),
        "home_status_model_ok": "Model ready",
        "home_status_model_no": "No model yet",
        "home_status_prauc": "PR-AUC",
        "home_status_data_ok": "Data cached",
        "home_status_data_no": "No cached data",
        "train_title": "Train ML Model",
        "train_expander": "Instructions",
        "train_expander_body": (
            "Train a classifier for “large drawdown within horizon” using only information available **before** each date.\n\n"
            "**Steps:** set model and dates → **Train Model** → review metrics on this page and under **Visualizations**."
        ),
        "train_sub_model": "Model configuration",
        "train_sub_data": "Data configuration",
        "train_model_type": "Model type",
        "train_calibration": "Calibration method",
        "train_horizon": "Prediction horizon (trading days)",
        "train_dd": "Drawdown threshold (%)",
        "train_start": "Training start",
        "train_end": "Training end",
        "test_start": "Test start",
        "test_end": "Test end",
        "data_start": "Data download start",
        "train_seed": "Random seed",
        "train_advanced": "Advanced settings",
        "train_n_est": "N estimators (RF / GB)",
        "train_max_depth": "Max depth (RF / GB)",
        "train_max_depth_unlimited": "Unlimited depth",
        "train_max_depth_help": "Uncheck to set a fixed max depth (can speed up / limit overfitting).",
        "train_button": "Train Model",
        "train_spinner": "Training model… This may take a few minutes.",
        "train_success": "Model trained successfully!",
        "train_fail": "Training failed",
        "live_title": "Live risk score",
        "live_warn": "Train a model first on **Train Model**.",
        "live_sub_get": "Run prediction",
        "live_sub_info": "Saved pipeline settings",
        "live_refresh": "Force fresh data download",
        "live_refresh_help": "Bypass cache; slower but closer to “right now.”",
        "live_button": "Get live risk score",
        "live_spinner": "Fetching data and scoring…",
        "live_horizon": "Prediction horizon",
        "live_threshold": "Drawdown threshold",
        "live_date": "As of date",
        "live_prob": "Crisis probability",
        "live_interp_title": "How to read this",
        "live_interp_body": (
            "There is a **{prob:.1%}** chance (model estimate) of at least **{dd:.1%}** drawdown within **{h}** trading days. "
            "Combine with fundamentals, liquidity needs, and your own risk limits. This is not advice."
        ),
        "live_fail": "Live prediction failed",
        "live_no_config": "No saved `pipeline_config.json` (train once to create it).",
        "test_title": "Upload and test custom data",
        "test_warn": "Train a model first on **Train Model**.",
        "test_intro": (
            "CSV needs **S&P 500** and **VIX** closes aligned by date. "
            "We accept common column names (see sample). Index or a **Date** column works."
        ),
        "test_sample_btn": "Download sample CSV template",
        "test_uploader": "Choose CSV file",
        "test_preview": "Data preview",
        "test_shape": "Shape",
        "test_cols": "Columns",
        "test_missing": "Missing required columns after normalization",
        "test_button": "Score uploaded data",
        "test_spinner": "Processing…",
        "test_success_pred": "Predictions computed",
        "test_results": "Prediction results",
        "test_download": "Download predictions CSV",
        "test_mean": "Mean probability",
        "test_max": "Max probability",
        "test_high_days": "Days ≥30% prob",
        "test_nan_warn": "Some features had NaN; forward-filled for scoring.",
        "test_read_fail": "Error reading file",
        "test_fail": "Scoring failed",
        "viz_title": "Model performance",
        "viz_warn": "Train a model first.",
        "viz_empty": "No figures yet. Train once to generate plots under `reports/figures/`.",
        "about_title": "About Crisis Radar",
        "about_body": (
            "### Scope\n"
            "Binary **drawdown-within-horizon** risk from ^GSPC and ^VIX features, time-safe splits, optional calibration.\n\n"
            "### Data\n"
            "yfinance: **^GSPC**, **^VIX**.\n\n"
            "### Docs\n"
            "See `README.md` for CLI, metrics, and limitations.\n\n"
            "### Credits\n"
            "Apostolos Chardalias | Finance Club UniPi — educational / research use."
        ),
        "common_rows": "rows",
        "decision_cheatsheet": "Member cheatsheet",
        "decision_cheatsheet_body": (
            "| Step | Action |\n"
            "|------|--------|\n"
            "| 1 | Train with dates that match the regime you care about |\n"
            "| 2 | Check **Brier** + calibration plot — well-calibrated probs are easier to interpret |\n"
            "| 3 | Use live score as **one** input; document horizon and threshold when you discuss |\n"
            "| 4 | On custom CSV, verify no gaps/duplicates in dates |\n"
            "| 5 | Never size positions from a single probability alone |"
        ),
    },
    "el": {
        "lang_label": "Γλώσσα",
        "nav_header": "Πλοήγηση",
        "page_home": "Αρχική",
        "page_train": "Εκπαίδευση μοντέλου",
        "page_live": "Ζωντανές προβλέψεις",
        "page_test": "Δοκιμή δεδομένων",
        "page_viz": "Οπτικοποιήσεις",
        "page_about": "Σχετικά",
        "sidebar_model_ready": "Το μοντέλο είναι εκπαιδευμένο και έτοιμο",
        "sidebar_no_model": "Δεν βρέθηκε εκπαιδευμένο μοντέλο. Εκπαιδεύστε πρώτα.",
        "sidebar_credits": "**Ανάπτυξη:** Apostolos Chardalias | Finance Club UniPi",
        "sidebar_version": "**Έκδοση:** 1.1.0",
        "home_title": "Καλώς ήρθατε στο Crisis Radar",
        "home_tagline": "Χρησιμοποιήστε τις <strong>βαθμονομημένες</strong> πιθανότητες για να <strong>ενισχύσετε</strong> την άποψή σας για τον κίνδυνο drawdown — όχι για να αντικαταστήσετε την κρίση σας.",
        "home_intro": (
            "Το **Crisis Radar** εκτιμά την πιθανότητα μεγάλου drawdown στον S&P 500 στα επόμενα *N* εργάσιμες, "
            "από ιστορικά στοιχεία, μεταβλητότητα και δείκτες τύπου VIX. Προορίζεται για μάθηση και έρευνα."
        ),
        "home_features_title": "Τι μπορείτε να κάνετε",
        "home_features": (
            "- **Εκπαίδευση** Random Forest, Gradient Boosting ή Λογιστική Παλινδρόμηση με χρονικό διαχωρισμό\n"
            "- **Ζωντανό σκορ** για την τελευταία συνεδρία (δεδομένα yfinance)\n"
            "- **Ανέβασμα CSV** για βαθμολόγηση δικής σας σειράς GSPC + VIX\n"
            "- **Γραφήματα** βαθμονόμησης, PR, confusion matrix, σημασία μεταβλητών"
        ),
        "home_qs_title": "Γρήγορα βήματα (πρώτη επίσκεψη)",
        "home_qs": (
            "1. **Εκπαίδευση μοντέλου** — προεπιλογές ή δικό σας ορίζοντα/κατώφλι, μετά **Εκπαίδευση** (μπορεί να πάρει λίγα λεπτά).\n"
            "2. **Ζωντανές προβλέψεις** — ενεργοποιήστε ανανέωση δεδομένων αν θέλετε τις τελευταίες τιμές.\n"
            "3. **Δοκιμή δεδομένων** — κατεβάστε το δείγμα CSV, αντικαταστήστε με δικά σας, ανεβάστε.\n"
            "4. **Οπτικοποιήσεις** — δείτε βαθμονόμηση και αν το μοντέλο σας ταιριάζει."
        ),
        "home_decision_title": "Πώς να βγάζετε καλύτερες αποφάσεις",
        "home_decision": (
            "- **Ζώνες κινδύνου**: ΧΑΜΗΛΟ &lt;10%, ΜΕΤΡΙΟ 10–30%, ΥΨΗΛΟ ≥30% (προεπιλογή — όχι σήμα συναλλαγών).\n"
            "- **Εμπιστοσύνη στη βαθμονόμηση**: αν η καμπύλη βαθμονόμησης είναι αδύναμη, αντιμετωπίστε τις πιθανότητες ως **σειρά κατάταξης**, όχι ακριβή %%.\n"
            "- **PR-AUC / F1**: χρήσιμα σε ανισόρροπες ετικέτες κρίσης· συγκρίνετε με αφελή baseline.\n"
            "- **Ορίζων & κατώφλι**: «−8%% σε 20 ημέρες» δεν είναι το ίδιο με «κραχ αύριο».\n"
            "- **Αποποίηση**: μόνο για εκπαίδευση/έρευνα — όχι επενδυτική συμβουλή."
        ),
        "home_status_model_ok": "Μοντέλο έτοιμο",
        "home_status_model_no": "Χωρίς μοντέλο",
        "home_status_prauc": "PR-AUC",
        "home_status_data_ok": "Δεδομένα στην cache",
        "home_status_data_no": "Χωρίς cache",
        "train_title": "Εκπαίδευση μοντέλου ML",
        "train_expander": "Οδηγίες",
        "train_expander_body": (
            "Εκπαίδευση ταξινομητή για «μεγάλο drawdown εντός ορίζοντα» μόνο με πληροφορίες διαθέσιμες **πριν** από κάθε ημερομηνία.\n\n"
            "**Βήματα:** ρυθμίσεις & ημερομηνίες → **Εκπαίδευση μοντέλου** → μετρικές εδώ και στις **Οπτικοποιήσεις**."
        ),
        "train_sub_model": "Ρυθμίσεις μοντέλου",
        "train_sub_data": "Ρυθμίσεις δεδομένων",
        "train_model_type": "Τύπος μοντέλου",
        "train_calibration": "Μέθοδος βαθμονόμησης",
        "train_horizon": "Ορίζων πρόβλεψης (εργάσιμες ημέρες)",
        "train_dd": "Κατώφλι drawdown (%)",
        "train_start": "Έναρξη εκπαίδευσης",
        "train_end": "Λήξη εκπαίδευσης",
        "test_start": "Έναρξη test",
        "test_end": "Λήξη test",
        "data_start": "Έναρξη λήψης δεδομένων",
        "train_seed": "Σπόρος τυχαιότητας",
        "train_advanced": "Προχωρημένα",
        "train_n_est": "N estimators (RF / GB)",
        "train_max_depth": "Μέγιστο βάθος (RF / GB)",
        "train_max_depth_unlimited": "Απεριόριστο βάθος",
        "train_max_depth_help": "Απενεργοποιήστε για σταθερό max depth (ταχύτερο / περιορισμός overfitting).",
        "train_button": "Εκπαίδευση μοντέλου",
        "train_spinner": "Εκπαίδευση… Μπορεί να διαρκέσει λίγα λεπτά.",
        "train_success": "Η εκπαίδευση ολοκληρώθηκε επιτυχώς!",
        "train_fail": "Η εκπαίδευση απέτυχε",
        "live_title": "Ζωντανό σκορ κινδύνου",
        "live_warn": "Εκπαιδεύστε πρώτα μοντέλο από **Εκπαίδευση μοντέλου**.",
        "live_sub_get": "Εκτέλεση πρόβλεψης",
        "live_sub_info": "Αποθηκευμένες ρυθμίσεις pipeline",
        "live_refresh": "Αναγκαστική λήψη φρέσκων δεδομένων",
        "live_refresh_help": "Χωρίς cache· πιο αργό αλλά πιο ενημερωμένο.",
        "live_button": "Υπολογισμός ζωντανού σκορ",
        "live_spinner": "Λήψη δεδομένων και υπολογισμός…",
        "live_horizon": "Ορίζων πρόβλεψης",
        "live_threshold": "Κατώφλι drawdown",
        "live_date": "Ημερομηνία αναφοράς",
        "live_prob": "Πιθανότητα κρίσης",
        "live_interp_title": "Πώς το διαβάζω",
        "live_interp_body": (
            "Εκτιμώμενη **{prob:.1%}** πιθανότητα για drawdown τουλάχιστον **{dd:.1%}** εντός **{h}** εργάσιμων. "
            "Συνδυάστε με fundamentals, ανάγκες ρευστότητας και όρια κινδύνου. Δεν είναι συμβουλή."
        ),
        "live_fail": "Αποτυχία ζωντανής πρόβλεψης",
        "live_no_config": "Δεν υπάρχει `pipeline_config.json` (εκπαιδεύστε μία φορά).",
        "test_title": "Ανέβασμα και δοκιμή δεδομένων",
        "test_warn": "Εκπαιδεύστε πρώτα μοντέλο από **Εκπαίδευση μοντέλου**.",
        "test_intro": (
            "Το CSV χρειάζεται κλείσιμα **S&P 500** και **VIX** ευθυγραμμισμένα ανά ημερομηνία. "
            "Δεχόμαστε συνήθη ονόματα στηλών (δείτε δείγμα). Index ή στήλη **Date**."
        ),
        "test_sample_btn": "Λήψη δείγματος CSV",
        "test_uploader": "Επιλογή αρχείου CSV",
        "test_preview": "Προεπισκόπηση",
        "test_shape": "Διαστάσεις",
        "test_cols": "Στήλες",
        "test_missing": "Λείπουν απαιτούμενες στήλες μετά την κανονικοποίηση",
        "test_button": "Βαθμολόγηση ανεβασμένων δεδομένων",
        "test_spinner": "Επεξεργασία…",
        "test_success_pred": "Οι προβλέψεις υπολογίστηκαν",
        "test_results": "Αποτελέσματα πρόβλεψης",
        "test_download": "Λήψη CSV προβλέψεων",
        "test_mean": "Μέση πιθανότητα",
        "test_max": "Μέγιστη πιθανότητα",
        "test_high_days": "Ημέρες με πιθαν. ≥30%",
        "test_nan_warn": "Κάποια χαρακτηριστικά είχαν NaN· εφαρμόστηκε forward-fill.",
        "test_read_fail": "Σφάλμα ανάγνωσης αρχείου",
        "test_fail": "Η βαθμολόγηση απέτυχε",
        "viz_title": "Απόδοση μοντέλου",
        "viz_warn": "Εκπαιδεύστε πρώτα μοντέλο.",
        "viz_empty": "Δεν υπάρχουν γραφήματα ακόμα. Εκπαιδεύστε μία φορά για `reports/figures/`.",
        "about_title": "Σχετικά με το Crisis Radar",
        "about_body": (
            "### Σκοπός\n"
            "Δυαδικό **drawdown-εντός-ορίζοντα** από χαρακτηριστικά ^GSPC και ^VIX, χρονικά ασφαλή διαχωρισμό, προαιρετική βαθμονόμηση.\n\n"
            "### Δεδομένα\n"
            "yfinance: **^GSPC**, **^VIX**.\n\n"
            "### Τεκμηρίωση\n"
            "Δείτε `README.md` για CLI, μετρικές, περιορισμούς.\n\n"
            "### Συντελεστές\n"
            "Apostolos Chardalias | Finance Club UniPi — εκπαιδευτική / ερευνητική χρήση."
        ),
        "common_rows": "γραμμές",
        "decision_cheatsheet": "Σύντομος οδηγός μελών",
        "decision_cheatsheet_body": (
            "| Βήμα | Ενέργεια |\n"
            "|------|----------|\n"
            "| 1 | Εκπαίδευση με ημερομηνίες που ταιριάζουν στο καθεστώς που σας ενδιαφέρει |\n"
            "| 2 | Ελέγξτε **Brier** + γράφημα βαθμονόμησης |\n"
            "| 3 | Το ζωντανό σκορ ως **ένα** εργαλείο· σημειώστε ορίζοντα και κατώφλι |\n"
            "| 4 | Σε δικό σας CSV, ελέγξτε κενά/διπλές ημερομηνίες |\n"
            "| 5 | Μην καθορίζετε θέσεις μόνο από μία πιθανότητα |"
        ),
    },
}


def t(lang: str, key: str, **kwargs: object) -> str:
    """Return translated string; `lang` is 'en' or 'el'."""
    lang = lang if lang in STRINGS else "en"
    bucket = STRINGS[lang]
    s = bucket.get(key) or STRINGS["en"].get(key) or key
    if kwargs:
        return s.format(**kwargs)
    return s


def sample_csv_bytes() -> bytes:
    """Small template CSV for members (GSPC_Close, VIX_Close)."""
    text = (
        "Date,GSPC_Close,VIX_Close\n"
        "2024-01-02,4740.56,12.35\n"
        "2024-01-03,4746.32,12.41\n"
        "2024-01-04,4720.78,13.02\n"
        "2024-01-05,4697.24,13.21\n"
        "2024-01-08,4763.21,13.08\n"
    )
    return text.encode("utf-8")
