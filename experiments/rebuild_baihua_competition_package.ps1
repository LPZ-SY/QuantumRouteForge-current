$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $RepoRoot ".venv312\Scripts\python.exe"

Push-Location $RepoRoot
try {
    & $Python experiments\analyze_quantum_candidate_quality.py `
        --inputs `
        results\baihua_hardware_p1_k64_seeds2_3_4_20260801 `
        results\baihua_hardware_p1_k64_seeds7_9_10_20260801 `
        --outdir results\baihua_quantum_candidate_quality_20260801

    & $Python experiments\analyze_baihua_candidate_budget.py `
        --inputs `
        results\baihua_hardware_pilot_p1_seed2_20260801 `
        results\baihua_hardware_pilot_p2_seeds1_2_20260801 `
        results\baihua_hardware_p1_seeds3_4_20260801 `
        results\baihua_hardware_p2_seeds3_4_20260801 `
        --outdir results\baihua_candidate_budget_20260801

    & $Python experiments\build_baihua_competition_report.py `
        --hardware-dirs `
        results\baihua_hardware_p1_k64_seeds2_3_4_20260801 `
        results\baihua_hardware_p1_k64_seeds7_9_10_20260801 `
        --candidate-budget-dir results\baihua_candidate_budget_20260801 `
        --ablation-dir results\baihua_ablation_suite_10seeds_20260801 `
        --headroom-screen results\baihua_headroom_screen_10seeds_20260801 `
        --legacy-p1-dirs `
        results\baihua_hardware_pilot_p1_seed2_20260801 `
        results\baihua_hardware_p1_seeds3_4_20260801 `
        --legacy-p2-dirs `
        results\baihua_hardware_pilot_p2_seeds1_2_20260801 `
        results\baihua_hardware_p2_seeds3_4_20260801 `
        --shenglian-width-scan results\cvrp_width_scan_20260731_final\width_scan_6_22.csv `
        --outdir results\baihua_competition_package_20260801
}
finally {
    Pop-Location
}
