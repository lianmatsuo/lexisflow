import streamlit as st
import subprocess
import sys
from pathlib import Path
import os
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="MIMIC-III Data Analysis",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
    .status-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# Title
st.markdown(
    '<h1 class="main-header">🏥 MIMIC-III Data Analysis Dashboard</h1>',
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.header("📊 Navigation")
    st.markdown("---")

    # Data source selection
    st.subheader("📁 Data Source Selection")

    mimic_dir = Path("data/mimic-iii-clinical-database-1.4")
    mimic_combined_path = mimic_dir / "combined" / "mimic_iii_combined.parquet"

    # Helper function to get best available file (Parquet preferred)
    def get_table_path(table_name):
        """Get table path, preferring Parquet over CSV."""
        parquet_path = mimic_dir / "parquet" / f"{table_name}.parquet"
        csv_path = mimic_dir / f"{table_name}.csv"
        if parquet_path.exists():
            return parquet_path
        elif csv_path.exists():
            return csv_path
        return None

    # Available tables (base names without extension)
    available_tables = {
        "ADMISSIONS": "ADMISSIONS",
        "PATIENTS": "PATIENTS",
        "ICUSTAYS": "ICUSTAYS",
        "SERVICES": "SERVICES",
        "TRANSFERS": "TRANSFERS",
        "DIAGNOSES_ICD": "DIAGNOSES_ICD",
        "PROCEDURES_ICD": "PROCEDURES_ICD",
        "CPTEVENTS": "CPTEVENTS",
        "DRGCODES": "DRGCODES",
        "PRESCRIPTIONS": "PRESCRIPTIONS",
        "CALLOUT": "CALLOUT",
        "LABEVENTS": "LABEVENTS",
        "OUTPUTEVENTS": "OUTPUTEVENTS",
        "MICROBIOLOGYEVENTS": "MICROBIOLOGYEVENTS",
    }

    # Check which tables exist (either Parquet or CSV)
    existing_tables = {
        name: base_name
        for name, base_name in available_tables.items()
        if get_table_path(base_name) is not None
    }

    # Combined table option
    use_combined = st.checkbox("Use Combined Table", value=mimic_combined_path.exists())

    if use_combined and mimic_combined_path.exists():
        st.success("✓ Combined dataset selected")
        selected_tables = []
    else:
        # Multi-select for individual tables
        selected_table_names = st.multiselect(
            "Select Tables",
            options=list(existing_tables.keys()),
            default=["ADMISSIONS"] if "ADMISSIONS" in existing_tables else [],
            help="Select one or more tables to analyze",
        )
        selected_tables = [
            get_table_path(existing_tables[name]) for name in selected_table_names
        ]
        selected_tables = [
            t for t in selected_tables if t is not None
        ]  # Filter None values

    # Store selection in session state
    st.session_state["use_combined"] = use_combined
    st.session_state["selected_tables"] = selected_tables
    st.session_state["mimic_combined_path"] = mimic_combined_path

    # Show selection summary
    if use_combined and mimic_combined_path.exists():
        st.info("📊 Selected: Combined Table")
    elif selected_tables:
        st.info(f"📊 Selected: {len(selected_tables)} table(s)")
        for table in selected_tables[:3]:  # Show first 3
            st.caption(f"  • {table.name}")
        if len(selected_tables) > 3:
            st.caption(f"  ... and {len(selected_tables) - 3} more")
    else:
        st.warning("⚠ No tables selected")

    st.markdown("---")

    # Output directories
    st.subheader("📂 Output Directories")
    correlation_dir = Path("eda/correlation_stats")
    anomaly_dir = Path("eda/anomaly_stats")

    if correlation_dir.exists():
        st.success(
            f"✓ Correlation stats: {len(list(correlation_dir.glob('*.png')))} files"
        )
    else:
        st.info("No correlation stats yet")

    if anomaly_dir.exists():
        st.success(f"✓ Anomaly stats: {len(list(anomaly_dir.glob('*.png')))} files")
    else:
        st.info("No anomaly stats yet")

# Main content tabs
tab1, tab2 = st.tabs(["📈 Correlation Statistics", "🔍 Anomaly Detection"])

# Tab 1: Correlation Statistics
with tab1:
    st.header("Correlation Statistics Analysis")
    st.markdown("""
    This analysis generates correlation matrices and association measures:
    - **Pearson, Spearman, Kendall** correlations for numeric variables
    - **Cramer's V** for categorical associations
    - **Theil's U** for asymmetric predictability
    - **ANOVA/Kruskal-Wallis** for categorical → numerical relationships
    """)

    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("Actions")
        run_correlation = st.button(
            "🚀 Run Correlation Analysis",
            type="primary",
            use_container_width=True,
            key="run_correlation",
        )

        if run_correlation:
            # Check if data is selected
            if (
                st.session_state.get("use_combined")
                and st.session_state.get("mimic_combined_path").exists()
            ):
                data_source = str(st.session_state["mimic_combined_path"])
            elif st.session_state.get("selected_tables"):
                data_source = st.session_state["selected_tables"]
            else:
                st.error("Please select at least one table or the combined table")
                st.stop()

            try:
                # Load and prepare data
                if isinstance(data_source, list):
                    # Multiple tables selected - combine them
                    dfs = []
                    for table_path in data_source:
                        path = Path(table_path)
                        # Prefer Parquet if available, fallback to CSV
                        if path.suffix == ".parquet":
                            df = pd.read_parquet(path)
                        else:
                            df = pd.read_csv(path, low_memory=False)
                        dfs.append(df)
                    # Simple merge on common columns (SUBJECT_ID, HADM_ID, etc.)
                    combined_df = dfs[0]
                    for df in dfs[1:]:
                        # Find common columns for merging
                        common_cols = set(combined_df.columns) & set(df.columns)
                        if "SUBJECT_ID" in common_cols and "HADM_ID" in common_cols:
                            combined_df = combined_df.merge(
                                df,
                                on=["SUBJECT_ID", "HADM_ID"],
                                how="outer",
                                suffixes=("", "_new"),
                            )
                        elif "SUBJECT_ID" in common_cols:
                            combined_df = combined_df.merge(
                                df, on="SUBJECT_ID", how="outer", suffixes=("", "_new")
                            )
                        else:
                            combined_df = pd.concat([combined_df, df], axis=1)

                    # Save temporary file
                    temp_file = Path("temp_correlation_data.parquet")
                    combined_df.to_parquet(temp_file)
                    data_source = str(temp_file)

                # Run the script with data source
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()

                status_text.info("🔄 Starting correlation analysis...")
                progress_bar.progress(10)

                # Run with unbuffered output
                env = os.environ.copy()
                env["PYTHONUNBUFFERED"] = "1"

                # Start subprocess with real-time output capture
                import threading
                import queue

                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-u",
                        "scripts/correlation_stats.py",
                        "--data",
                        data_source,
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=os.getcwd(),
                    env=env,
                    bufsize=1,
                )

                # Track progress based on output
                stdout_lines = []
                stderr_lines = []
                progress_steps = {"numeric": 30, "cramer": 50, "theil": 70, "anova": 90}

                output_queue = queue.Queue()

                def read_output(pipe, queue, is_stderr=False):
                    for line in iter(pipe.readline, ""):
                        queue.put(("stderr" if is_stderr else "stdout", line))
                    pipe.close()

                stdout_thread = threading.Thread(
                    target=read_output, args=(process.stdout, output_queue)
                )
                stderr_thread = threading.Thread(
                    target=read_output, args=(process.stderr, output_queue, True)
                )
                stdout_thread.start()
                stderr_thread.start()

                # Update progress based on output
                while process.poll() is None:
                    try:
                        source, line = output_queue.get(timeout=0.1)
                        if source == "stdout":
                            stdout_lines.append(line)
                            # Update progress based on keywords
                            line_lower = line.lower()
                            line_stripped = line.strip()

                            # Show detailed progress messages
                            if "creating visualization" in line_lower:
                                # Extract visualization name
                                if "numeric correlations" in line_lower:
                                    progress_bar.progress(progress_steps["numeric"])
                                    status_text.info(
                                        "🎨 Creating visualization: Numeric Correlations (Pearson, Spearman, Kendall)"
                                    )
                                elif "cramer" in line_lower:
                                    progress_bar.progress(progress_steps["cramer"])
                                    status_text.info(
                                        "🎨 Creating visualization: Cramer's V Matrix"
                                    )
                                elif "theil" in line_lower:
                                    progress_bar.progress(progress_steps["theil"])
                                    status_text.info(
                                        "🎨 Creating visualization: Theil's U Matrix"
                                    )
                                elif "anova" in line_lower or "kruskal" in line_lower:
                                    progress_bar.progress(progress_steps["anova"])
                                    status_text.info(
                                        "🎨 Creating visualization: ANOVA/Kruskal-Wallis Matrix"
                                    )
                            elif "drawing" in line_lower or "plotting" in line_lower:
                                # Show sub-steps
                                if "pearson" in line_lower:
                                    status_text.info(
                                        "  → Drawing Pearson correlation heatmap..."
                                    )
                                elif "spearman" in line_lower:
                                    status_text.info(
                                        "  → Drawing Spearman correlation heatmap..."
                                    )
                                elif "kendall" in line_lower:
                                    status_text.info(
                                        "  → Drawing Kendall correlation heatmap..."
                                    )
                                elif "cramer" in line_lower:
                                    status_text.info(
                                        "  → Drawing Cramer's V heatmap..."
                                    )
                                elif "theil" in line_lower:
                                    status_text.info("  → Drawing Theil's U heatmap...")
                                elif "anova f-statistics" in line_lower:
                                    status_text.info(
                                        "  → Drawing ANOVA F-statistics heatmap..."
                                    )
                                elif "anova p-values" in line_lower:
                                    status_text.info(
                                        "  → Drawing ANOVA p-values heatmap..."
                                    )
                                elif "kruskal-wallis h-statistics" in line_lower:
                                    status_text.info(
                                        "  → Drawing Kruskal-Wallis H-statistics heatmap..."
                                    )
                                elif "kruskal-wallis p-values" in line_lower:
                                    status_text.info(
                                        "  → Drawing Kruskal-Wallis p-values heatmap..."
                                    )
                                elif "histogram" in line_lower:
                                    status_text.info("  → Drawing histogram...")
                                elif "heatmap" in line_lower:
                                    status_text.info("  → Drawing heatmap...")
                                else:
                                    # Show the actual message
                                    status_text.info(f"  → {line_stripped}")
                            elif "saving visualization" in line_lower:
                                status_text.info("  → Saving visualization to file...")
                            elif "saved visualization" in line_lower:
                                # Extract filename
                                if "numeric_correlations" in line_lower:
                                    progress_bar.progress(progress_steps["numeric"] + 5)
                                    status_text.info(
                                        "✓ Saved: numeric_correlations.png"
                                    )
                                elif "cramers_v_matrix" in line_lower:
                                    progress_bar.progress(progress_steps["cramer"] + 5)
                                    status_text.info("✓ Saved: cramers_v_matrix.png")
                                elif "theils_u_matrix" in line_lower:
                                    progress_bar.progress(progress_steps["theil"] + 5)
                                    status_text.info("✓ Saved: theils_u_matrix.png")
                                elif "anova_kruskal" in line_lower:
                                    progress_bar.progress(progress_steps["anova"] + 5)
                                    status_text.info(
                                        "✓ Saved: anova_kruskal_matrix.png"
                                    )
                            elif "calculating numeric correlations" in line_lower:
                                progress_bar.progress(progress_steps["numeric"] - 10)
                                status_text.info(
                                    "📊 Calculating numeric correlations..."
                                )
                            elif "calculating cramer" in line_lower:
                                progress_bar.progress(progress_steps["cramer"] - 10)
                                status_text.info("📊 Calculating Cramer's V matrix...")
                            elif "calculating theil" in line_lower:
                                progress_bar.progress(progress_steps["theil"] - 10)
                                if "vs all categories" in line_lower:
                                    parts = line_stripped.split(":")
                                    if len(parts) > 1:
                                        category_info = parts[1].split("vs")[0].strip()
                                        status_text.info(
                                            f"📊 Calculating Theil's U: {category_info}..."
                                        )
                                    else:
                                        status_text.info(
                                            "📊 Calculating Theil's U matrix..."
                                        )
                                else:
                                    status_text.info(
                                        "📊 Calculating Theil's U matrix..."
                                    )
                            elif "calculating anova" in line_lower:
                                progress_bar.progress(progress_steps["anova"] - 10)
                                if "vs all numerical" in line_lower:
                                    parts = line_stripped.split(":")
                                    if len(parts) > 1:
                                        category_info = parts[1].split("vs")[0].strip()
                                        status_text.info(
                                            f"📊 Calculating ANOVA/Kruskal-Wallis: {category_info}..."
                                        )
                                    else:
                                        status_text.info(
                                            "📊 Calculating ANOVA/Kruskal-Wallis..."
                                        )
                                else:
                                    status_text.info(
                                        "📊 Calculating ANOVA/Kruskal-Wallis statistics..."
                                    )
                        else:
                            stderr_lines.append(line)
                    except queue.Empty:
                        continue

                # Wait for threads to finish
                stdout_thread.join(timeout=1)
                stderr_thread.join(timeout=1)

                # Get remaining output
                stdout, stderr = process.communicate()
                stdout_lines.extend(stdout.splitlines() if stdout else [])
                stderr_lines.extend(stderr.splitlines() if stderr else [])

                result = type(
                    "obj",
                    (object,),
                    {
                        "returncode": process.returncode,
                        "stdout": "\n".join(stdout_lines),
                        "stderr": "\n".join(stderr_lines),
                    },
                )()

                # Clean up temp file if created
                if Path("temp_correlation_data.parquet").exists():
                    Path("temp_correlation_data.parquet").unlink()

                progress_bar.progress(100)
                status_text.empty()

                if result.returncode == 0:
                    st.success("✓ Analysis completed successfully!")
                    # Show last few lines of output
                    if result.stdout:
                        output_lines = result.stdout.strip().split("\n")
                        if len(output_lines) > 5:
                            st.text("\n".join(output_lines[-5:]))
                        else:
                            st.text(result.stdout)
                    st.rerun()  # Refresh to show new visualizations
                else:
                    st.error("✗ Analysis failed")
                    if result.stderr:
                        st.text(result.stderr)
                    if result.stdout:
                        st.text(result.stdout)
            except subprocess.TimeoutExpired:
                if "progress_bar" in locals():
                    progress_bar.progress(100)
                if "status_text" in locals():
                    status_text.empty()
                st.error("✗ Analysis timed out after 10 minutes")
            except Exception as e:
                if "progress_bar" in locals():
                    progress_bar.progress(100)
                if "status_text" in locals():
                    status_text.empty()
                st.error(f"Error: {e}")
                import traceback

                st.text(traceback.format_exc())

        # Refresh button
        if st.button(
            "🔄 Refresh Visualizations",
            use_container_width=True,
            key="refresh_correlation",
        ):
            st.rerun()

    with col2:
        st.subheader("Visualizations")

        # Check for generated files
        correlation_files = {
            "Numeric Correlations": "numeric_correlations.png",
            "Cramer's V Matrix": "cramers_v_matrix.png",
            "Theil's U Matrix": "theils_u_matrix.png",
            "ANOVA/Kruskal-Wallis": "anova_kruskal_matrix.png",
        }

        for viz_name, filename in correlation_files.items():
            filepath = correlation_dir / filename
            if filepath.exists():
                st.subheader(viz_name)
                st.image(str(filepath), use_container_width=True)
                st.markdown("---")
            else:
                st.info(
                    f"📊 {viz_name} not yet generated. Run the analysis to create it."
                )

# Tab 2: Anomaly Detection
with tab2:
    st.header("Anomaly Detection & Statistical Analysis")
    st.markdown("""
    This analysis performs comprehensive statistical analysis and anomaly detection:
    - **Z-scores** and distributions
    - **IQR** outlier detection
    - **Quantile ranks**
    - **Mahalanobis distance** (multivariate outliers)
    - **kNN distance**
    - **Anomaly detectors**: Isolation Forest, LOF, One-Class SVM
    - **Joint distributions** and mutual information
    """)

    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("Actions")
        run_anomaly = st.button(
            "🚀 Run Anomaly Detection",
            type="primary",
            use_container_width=True,
            key="run_anomaly",
        )

        if run_anomaly:
            # Check if data is selected
            if (
                st.session_state.get("use_combined")
                and st.session_state.get("mimic_combined_path").exists()
            ):
                data_source = str(st.session_state["mimic_combined_path"])
            elif st.session_state.get("selected_tables"):
                data_source = st.session_state["selected_tables"]
            else:
                st.error("Please select at least one table or the combined table")
                st.stop()

            try:
                # Load and prepare data
                if isinstance(data_source, list):
                    # Multiple tables selected - combine them
                    dfs = []
                    for table_path in data_source:
                        path = Path(table_path)
                        # Prefer Parquet if available, fallback to CSV
                        if path.suffix == ".parquet":
                            df = pd.read_parquet(path)
                        else:
                            df = pd.read_csv(path, low_memory=False)
                        dfs.append(df)
                    # Simple merge on common columns
                    combined_df = dfs[0]
                    for df in dfs[1:]:
                        common_cols = set(combined_df.columns) & set(df.columns)
                        if "SUBJECT_ID" in common_cols and "HADM_ID" in common_cols:
                            combined_df = combined_df.merge(
                                df,
                                on=["SUBJECT_ID", "HADM_ID"],
                                how="outer",
                                suffixes=("", "_new"),
                            )
                        elif "SUBJECT_ID" in common_cols:
                            combined_df = combined_df.merge(
                                df, on="SUBJECT_ID", how="outer", suffixes=("", "_new")
                            )
                        else:
                            combined_df = pd.concat([combined_df, df], axis=1)

                    # Save temporary file
                    temp_file = Path("temp_anomaly_data.parquet")
                    combined_df.to_parquet(temp_file)
                    data_source = str(temp_file)

                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()

                status_text.info("🔄 Starting anomaly detection analysis...")
                progress_bar.progress(5)

                # Run with unbuffered output
                env = os.environ.copy()
                env["PYTHONUNBUFFERED"] = "1"

                # Start subprocess with real-time output capture
                import threading
                import queue

                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-u",
                        "scripts/anomaly_detection.py",
                        "--data",
                        data_source,
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=os.getcwd(),
                    env=env,
                    bufsize=1,
                )

                # Track progress based on output
                stdout_lines = []
                stderr_lines = []
                progress_steps = {
                    "z-score": 10,
                    "iqr": 20,
                    "quantile": 30,
                    "mahalanobis": 40,
                    "knn": 50,
                    "anomaly": 70,
                    "rule": 80,
                    "joint": 90,
                }

                # Read output line by line
                output_queue = queue.Queue()

                def read_output(pipe, queue, is_stderr=False):
                    for line in iter(pipe.readline, ""):
                        queue.put(("stderr" if is_stderr else "stdout", line))
                    pipe.close()

                stdout_thread = threading.Thread(
                    target=read_output, args=(process.stdout, output_queue)
                )
                stderr_thread = threading.Thread(
                    target=read_output, args=(process.stderr, output_queue, True)
                )
                stdout_thread.start()
                stderr_thread.start()

                # Update progress based on output
                while process.poll() is None:
                    try:
                        source, line = output_queue.get(timeout=0.1)
                        if source == "stdout":
                            stdout_lines.append(line)
                            # Update progress based on keywords
                            line_lower = line.lower()
                            if (
                                "z-score" in line_lower
                                or "calculating z-score" in line_lower
                            ):
                                progress_bar.progress(progress_steps["z-score"])
                                status_text.info("📊 Calculating Z-scores...")
                            elif "iqr" in line_lower and "calculating" in line_lower:
                                progress_bar.progress(progress_steps["iqr"])
                                status_text.info("📊 Calculating IQR...")
                            elif (
                                "quantile" in line_lower and "calculating" in line_lower
                            ):
                                progress_bar.progress(progress_steps["quantile"])
                                status_text.info("📊 Calculating quantile ranks...")
                            elif "mahalanobis" in line_lower:
                                progress_bar.progress(progress_steps["mahalanobis"])
                                status_text.info(
                                    "📊 Calculating Mahalanobis distance..."
                                )
                            elif "knn" in line_lower or "k-nearest" in line_lower:
                                progress_bar.progress(progress_steps["knn"])
                                status_text.info("📊 Calculating kNN distance...")
                            elif (
                                "anomaly detector" in line_lower
                                or "isolation" in line_lower
                                or "lof" in line_lower
                                or "one-class" in line_lower
                            ):
                                progress_bar.progress(progress_steps["anomaly"])
                                status_text.info("📊 Running anomaly detectors...")
                            elif "rule violation" in line_lower:
                                progress_bar.progress(progress_steps["rule"])
                                status_text.info("📊 Checking rule violations...")
                            elif (
                                "joint distribution" in line_lower
                                or "mutual information" in line_lower
                            ):
                                progress_bar.progress(progress_steps["joint"])
                                status_text.info("📊 Measuring joint distributions...")
                        else:
                            stderr_lines.append(line)
                    except queue.Empty:
                        continue

                # Wait for threads to finish
                stdout_thread.join(timeout=1)
                stderr_thread.join(timeout=1)

                # Get remaining output
                stdout, stderr = process.communicate()
                stdout_lines.extend(stdout.splitlines() if stdout else [])
                stderr_lines.extend(stderr.splitlines() if stderr else [])

                result = type(
                    "obj",
                    (object,),
                    {
                        "returncode": process.returncode,
                        "stdout": "\n".join(stdout_lines),
                        "stderr": "\n".join(stderr_lines),
                    },
                )()

                # Clean up temp file if created
                if Path("temp_anomaly_data.parquet").exists():
                    Path("temp_anomaly_data.parquet").unlink()

                progress_bar.progress(100)
                status_text.empty()

                if result.returncode == 0:
                    st.success("✓ Analysis completed successfully!")
                    # Show last few lines of output
                    if result.stdout:
                        output_lines = result.stdout.split("\n")
                        if len(output_lines) > 10:
                            st.text("\n".join(output_lines[-10:]))
                        else:
                            st.text(result.stdout)
                    st.rerun()  # Refresh to show new visualizations
                else:
                    st.error("✗ Analysis failed")
                    if result.stderr:
                        st.text(result.stderr)
                    if result.stdout:
                        st.text(result.stdout)
            except subprocess.TimeoutExpired:
                if "progress_bar" in locals():
                    progress_bar.progress(100)
                if "status_text" in locals():
                    status_text.empty()
                st.error("✗ Analysis timed out after 30 minutes")
            except Exception as e:
                if "progress_bar" in locals():
                    progress_bar.progress(100)
                if "status_text" in locals():
                    status_text.empty()
                st.error(f"Error: {e}")
                import traceback

                st.text(traceback.format_exc())

        # Refresh button
        if st.button(
            "🔄 Refresh Visualizations", use_container_width=True, key="refresh_anomaly"
        ):
            st.rerun()

    with col2:
        st.subheader("Visualizations")

        # Check for generated files
        anomaly_files = {
            "Z-Score Distributions": "z_scores_distributions.png",
            "IQR Outliers": "iqr_outliers.png",
            "Quantile Ranks": "quantile_ranks_distributions.png",
            "Mahalanobis Distance": "mahalanobis_distance.png",
            "kNN Distance": "knn_distance.png",
            "Anomaly Detectors": "anomaly_detectors.png",
            "Joint Distributions": "joint_distributions.png",
            "Mutual Information Matrix": "mutual_information_matrix.png",
        }

        # Group visualizations in expandable sections
        for viz_name, filename in anomaly_files.items():
            filepath = anomaly_dir / filename
            if filepath.exists():
                with st.expander(f"📊 {viz_name}", expanded=True):
                    st.image(str(filepath), use_container_width=True)
            else:
                st.info(
                    f"📊 {viz_name} not yet generated. Run the analysis to create it."
                )
                st.markdown("---")

# Footer
st.markdown("---")
st.markdown(
    """
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p>MIMIC-III Data Analysis Dashboard</p>
    <p><small>Generated visualizations are saved in <code>eda/</code> directory</small></p>
</div>
""",
    unsafe_allow_html=True,
)
