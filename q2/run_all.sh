PY=../venv/Scripts/python.exe
for m in B C; do for s in 0 1; do $PY train.py --model $m --seed $s > train${m}_s$s.log 2>&1; done; done
$PY choose_d.py > choose_d.log 2>&1
for s in 0 1; do $PY train.py --model D --seed $s > trainD_s$s.log 2>&1; done
echo done > run_all.done
