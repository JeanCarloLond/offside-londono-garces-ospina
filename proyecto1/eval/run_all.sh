set -e
echo "=== 1/3 reentrenando LoRA sin el eval set ==="
python train_holdout_model.py > train_log.txt 2>&1
grep -E "Corpus:|train=|eval final" train_log.txt
echo "=== 2/3 scorecard completo (3 sistemas, con juez) ==="
python harness.py --adapter ../m1_lora_adapter_holdout > scorecard_run.log 2>&1
sed -n '/SCORECARD/,$p' scorecard_run.log
echo "=== 3/3 experimento de sesgo de verbosidad ==="
python bias_check.py > bias_run.log 2>&1
sed -n '/SESGO DE VERBOSIDAD/,$p' bias_run.log
echo "=== TODO LISTO ==="
