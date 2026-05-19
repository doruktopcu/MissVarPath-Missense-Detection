$ErrorActionPreference = "Continue"
$PY = "C:\Users\Doruk-Topcu\anaconda3\python.exe"

$VEP = @(
    "alphamissense_", "cadd_", "cadd_exome_", "revel_", "sift_",
    "metarnn_", "bayesdel_", "fathmm_", "mutationtaster_", "provean_",
    "esm1b_", "eve_", "primateai_", "mvp_", "ditto_",
    "mutation_assessor_", "vest_", "chasmplus", "mutpred1_", "aloft_", "gmvp_",
    "polyphen2_", "gerp_", "phastcons_", "phylop_", "siphy_"
)

$RAW = @(
    "alfa_", "allofus250k_", "gnomad_", "gnomad3_", "regeneron_",
    "hg19_pos", "original_input_pos", "kmer_", "gc_", "entropy_"
)

$MODELS = @("KNN", "NearestCentroid", "CosineSimilarity", "DecisionTree",
            "LDA", "QDA", "LinearSVC", "RidgeClassifier", "SGDClassifier",
            "HistGradientBoosting")

function Stamp { (Get-Date -Format "HH:mm:ss") }

function Run-It {
    param([string]$Name, [string[]]$ExtraArgs)
    $log = "outputs/reports/train_$Name.log"
    Write-Host ""
    Write-Host "===== [$(Stamp)] START $Name ====="
    $args = $ExtraArgs + @("--models") + $MODELS
    & $PY -m src.train @args *>&1 | Out-File -Encoding utf8 $log
    Write-Host "===== [$(Stamp)] DONE  $Name ====="
}

Run-It "4class_gene"               @("--task","4class","--cv-mode","gene","--tag","gene")
Run-It "2class_gene"               @("--task","2class","--cv-mode","gene","--tag","gene")
Run-It "4class_augmented"          @("--task","4class","--variant","augmented","--tag","augmented")
Run-It "2class_augmented"          @("--task","2class","--variant","augmented","--tag","augmented")

$rawArgs4 = @("--task","4class","--variant","augmented","--keep-prefixes") + $RAW + @("--tag","raw_only")
$rawArgs2 = @("--task","2class","--variant","augmented","--keep-prefixes") + $RAW + @("--tag","raw_only")
Run-It "4class_augmented_raw_only" $rawArgs4
Run-It "2class_augmented_raw_only" $rawArgs2

$noVepArgs4 = @("--task","4class","--variant","augmented","--drop-prefixes") + $VEP + @("--tag","no_vep")
$noVepArgs2 = @("--task","2class","--variant","augmented","--drop-prefixes") + $VEP + @("--tag","no_vep")
$noVepGene4 = @("--task","4class","--variant","augmented","--cv-mode","gene","--drop-prefixes") + $VEP + @("--tag","no_vep")
Run-It "4class_augmented_no_vep"      $noVepArgs4
Run-It "2class_augmented_no_vep"      $noVepArgs2
Run-It "4class_augmented_gene_no_vep" $noVepGene4

Run-It "4class_no_ditto" @("--task","4class","--drop-prefixes","ditto_","--tag","no_ditto")

Write-Host ""
Write-Host "===== [$(Stamp)] ALL EXPERIMENTS COMPLETE ====="
