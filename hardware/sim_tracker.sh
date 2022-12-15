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

# check if we have a directory to save the results of the tracker
TRACKER_DIR=tracker_results
if [ -d "$TRACKER_DIR" ];
then
  echo "Directory $TRACKER_DIR already exists."
else 
  echo "Creating directory $TRACKER_DIR"
  mkdir $TRACKER_DIR
fi

# check if a directory with the name of the current app exists
APP_DIR=$TRACKER_DIR/${cur_app}
if [ -d "$APP_DIR" ];
then
  echo "Directory $APP_DIR already exists."
else 
  echo "Creating directory $APP_DIR"
  mkdir -p $APP_DIR
fi

idx=$RANDOM
file_name="tracker_${cur_app}_${idx}.txt"
file_path=${APP_DIR}/${file_name}

if [ -f "$file_path" ];
then 
  new_rand=$RANDOM
  idx = $(( idx + new_rand ))
  file_name="tracker_${cur_app}_${idx}.txt"
fi 

START_D=$( date "+%d/%m/%y" )
START_H=$( date "+%H:%M:%S" )
echo "[MemPool HW] Simulation started at: $START_H on $START_D"
echo "Saving output to: ${file_path}"
start_time=$SECONDS
# sleep 2
app=${cur_app} make simcvcs #2>&1 | tee ${file_path}
elapsed=$(( SECONDS - start_time ))
END_D=$( date "+%d/%m/%y" )
END_H=$( date "+%H:%M:%S" )
echo "[MemPool HW] Simulation finished at: $END_H on $END_D"
eval "echo [MemPool HW] Elapsed time: $(date -ud "@$elapsed" +'%H hr %M min %S sec')"

echo "Generating the traces for run id: ${idx}."
make trace 2>&1 | tee ${file_path}
result_id=$(grep -oP '(?<= tee results/).*(?=/)' ${file_path})
mail -s "[MemPool] Finished run with id ${idx}" vivianep@iis.ee.ethz.ch <<< "File stored in: ${file_path}. Simulation took $(date -ud "@$elapsed" +'$((%s/3600/24)) days %H hr %M min %S sec') . The results are stored in: /result/${result_id}."
echo "[MemPool HW] Message sent."

# grep name of directory name after */hardware/results/ in the text file
# and store it in a variable

