set -e 
#set -x


for WIDTH in 16 64 256; do
	export path="exp_results/ets_${WIDTH}_math500/"
			
	python3 ./math_evaluate.py   --path $path/answers.json \
	    --agg_func majority_vote \
	    --output_path $path/results_vote_last.txt \
	    --model_type llemma \
	    --weighted True \
	    --weight_agg last
done

