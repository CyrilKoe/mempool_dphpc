while getopts :a:h opt; do
  case "${opt}" in
    h)
      echo "[MemPool HW] Usage: sim_build.sh [-h]"
      echo "[MemPool HW]   -h: Display this help message"
      echo "[MemPool HW] Usage: sim_build.sh [-a]"
      echo "[MemPool HW]   -a: Select which app you want to run."
      exit 1
      ;;
    a)
      cur_app="${OPTARG}"
      echo "[MemPool HW] Simulating application $cur_app."
      ;;
    \?)
      echo "[MemPool HW] Invalid option: -$OPTARG" >&2
      exit 1
      ;;
  esac
done

START_D=$( date "+%d/%m/%y" )
START_H=$( date "+%H:%M:%S" )
echo "[MemPool HW] Simulation started at: $START_H on $START_D"
start_time=$SECONDS
# sleep 2
app=pooling make simcvcs
elapsed=$(( SECONDS - start_time ))
END_D=$( date "+%d/%m/%y" )
END_H=$( date "+%H:%M:%S" )
echo "[MemPool HW] Simulation finished at: $END_H on $END_D"

eval "echo [MemPool HW] Elapsed time: $(date -ud "@$elapsed" +'$((%s/3600/24)) days %H hr %M min %S sec')"