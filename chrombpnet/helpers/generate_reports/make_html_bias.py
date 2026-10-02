import json
import pandas as pd
import os
import argparse
from chrombpnet.helpers.generate_reports import report_shell
from chrombpnet.helpers.generate_reports import modisco_table

def read_args():
	parser = argparse.ArgumentParser(description="Make summary reports")
	parser.add_argument('-id', '--input-dir', type=str, help="directory name output by command chrombpnet bias pipeline")
	parser.add_argument('-fp', '--file-prefix', required=False, default=None, type=str, help="File prefix for output to use. All the files will be prefixed with this string if provided.")
	parser.add_argument('-icmd', '--command', required=False, default="pipeline", type=str, choices=['pipeline', 'train', "qc"], help="Choices on what to include in the report - entire pipleline, only train, only qc")
	parser.add_argument('-hp', '--html_prefix', required=False, default="./", help="The html prefix to use for the html file output.")

	args = parser.parse_args()
	return args

def train_report(fpx,prefix):
	# preprocessing defaults
	pre_hed = 'Preprocessing report'
	pre_text = 'The image below should look closely like a Tn5 or DNase bias enzyme motif. '

	bias_image_loc=os.path.join("./","{}bw_shift_qc.png".format(fpx))

	## training images

	train_hed = 'Training report'
	train_text = 'The val loss (validation loss) will decrease and saturate after a few epochs.'

	loss_image_loc = report_shell.loss_curve(prefix, fpx, "bias.log")

	html = f'''
			<body style="font-size:22px;">
				<h3>{pre_hed}</h3>
				<p>{pre_text}</p>
				<img src={{bias_image}} style="max-width: 60%;">
			</body>
			<body style="font-size:22px;">
				<h3>{train_hed}</h3>			
				<p>{train_text}</p>
				<img src={{loss_image_loc}} class="center", style="max-width: 20%">
			</body>
			'''
			
	return html.format(bias_image=bias_image_loc,loss_image_loc=loss_image_loc)
	
def qc_report(fpx,prefix):
	## bias model training performance

	bias_model_perf_hed = 'Bias model performance in peaks and non-peaks'
	bias_model_perf_text = '<b> Counts Metrics: </b> The pearsonr in non-peaks should be greater than 0 (higher the better). \
	The pearsonr in peaks should be greater than -0.3 (otherwise the bias model could potentially be capturing AT bias). \
	MSE (Mean Squared Error) will be high in peaks. <br> <br> <b> Profile Metrics:</b>  Median JSD (Jensen Shannon Divergence between observed and predicted) lower the better. \
	Median norm JSD is median of the min-max normalized JSD where min JSD is the worst case JSD i.e JSD of observed with uniform profile and \
	max JSD is the best case JSD i.e 0. Median norm JSD is higher the better. Both JSD and median norm JSD are sensitive to read-depth. \
	Higher read-depth results in better metrics. \
	<br> \
	<br> \
	<b> What to do if your pearsonr in peaks is less than -0.3? </b> \
	In the range of -0.3 to -0.5 please be wary of your chrombpnet_nobias.pt (that will potentially be trained with this bias model) TFModisco showing lots of GC rich motifs (> 3 in the top-10). If this is not the case \
	you can continue using the chrombpnet_nobias.pt. If you end up seeing a lot of GC rich motifs it is likely that bias model has learnt a different GC distribution than your GC-content in peaks. \
	You might benefit from increasing the bias_threshold_factor argument input to the <em>chrombpnet bias pipeline</em> or <em>chrombpnet bias train</em> command used in training the bias model \
	and retrain a new bias model. For more intuition about this argument refer to the <a href=\"https://github.com/kundajelab/chrombpnet/wiki/FAQ\">FAQ</a> section in wiki. \
	If the value is less than -0.5 the <a href=\"https://github.com/kundajelab/chrombpnet/wiki/ChromBPNet-training\">chrombpnet training</a> will automatically throw an error.'
	

	data = json.load(open(os.path.join(prefix,"evaluation/{}bias_metrics.json".format(fpx))))
	df = report_shell.metrics_table(data, 'counts_metrics', 'counts_metrics')

	df1 = report_shell.metrics_table(data, 'profile_metrics', 'profile_metrics')

	## TFModisco motifs learnt by bias model (bias.pt) model 
	
	def remove_negs(tables):
		new_lines=[]
		set_flag = True
		lines = tables.split("\n")
		jdx=0
		for idx in range(len(lines)-1):
			
			if jdx==15:
				set_flag = True
				jdx=0
	
			if "neg_" in lines[idx+1]:
				set_flag=False
				jdx = 0
		
			if set_flag:
				new_lines.append(lines[idx])
			else:
				jdx+=1
		new_lines.append(lines[-1])
		return  "\n".join(new_lines)
				
		
	tf_hed = "TFModisco motifs learnt from bias model (bias.pt) model"
	
	
	
	tf_text_profile = "<b> TFModisco motifs generated from profile contribution scores of the bias model. </b> \
	cwm_fwd, cwm_rev are the forward and reverse complemented consolidated motifs from contribution scores in subset of random peaks. \
	These CWM motifs should be free from any Transcription Factor (TF) motifs and should contain either only bias motifs or random repeats.\
	For each of these motifs, we use TOMTOM to find the top-3 closest matches (match_0, match_1, match_2) from a database consisting of both \
	MEME TF motifs and heterogenous enzyme bias motifs that we have repeatedly seen in our datasets.  \
	The qvals (qval0,qval1,qval2) should be high (> 0.0001) if the closest hit is a TF motif (i.e indicating that the closest match is not the correct match) - this is also generally \
	verifiable by eye as the closest match will look nothing like the CWMs. The qvals should be low if the closest hit is enzyme bias motif and \
	generally verifiable that the top match looks like the CWM. The first 3-5 motifs in the list below should look like enzyme bias motif. \
	<br> \
	<br> \
	<b> What to do if you find an obvious TF motif in the list? </b> <br> \
	Do not use this bias model as it will regress the contribution of the TF motifs (along with bias motifs) from the chrombpnet_nobias.pt. Reduce the bias_threshold_factor argument \
	input to the <em>chrombpnet bias pipeline</em> or <em>chrombpnet bias train</em> command used in training the bias model and retrain a new bias model. For more intuition about this argument \
	refer to the <a href=\"https://github.com/kundajelab/chrombpnet/wiki/FAQ\">FAQ</a> section in wiki. \
	<br> \
	<br> \
	<b> What to do if you are unsure if a given CWM motif is resembling the match_0 logo for example? </b> <br> \
	Get marginal footprint on the match_0 motif logo (using the command <em>chrombpnet footprints</em> and make sure that the bias models footprint is closer to that of controls with no motif \
	inserted - for examples look at <a href=\"https://github.com/kundajelab/chrombpnet/wiki/FAQ\">FAQ</a> )"

	tf_text_counts = "<b> TFModisco motifs generated from counts contribution scores of the bias model. </b> \
	cwm_fwd, cwm_rev are the forward and reverse complemented consolidated motifs from contribution scores in subset of random peaks. \
	These motifs should be free from any Transcription Factor (TF) motifs and should contain motifs either weakly related to bias motifs or random repeats. \
	For each of these motifs, we use TOMTOM to find the top-3 closest matches (match_0, match_1, match_2) from a database consisting of both \
	MEME TF motifs and heterogenous enzyme bias motifs that we have repeatedly seen in our datasets.  \
	The qvals should be high (> 0.0001) if the closest hit is a TF motif (i.e indicating that the closest match is not the correct match, this is also generally \
	verifiable by eye and making sure the closest match looks nothing like the CWMs). \
	<br> \
	<br> \
	<b> What to do if you find an obvious TF motif in the list? </b> <br> \
	Do not use this bias model as it will regress the contribution of the TF motifs (along with bias motifs) from the chrombpnet_nobias.pt. Reduce the bias_threshold_factor argument \
	input to the <em>chrombpnet bias pipeline</em> or <em>chrombpnet bias train</em> command used in training the bias model and retrain a new bias model. For more intuition about this argument \
	refer to the <a href=\"https://github.com/kundajelab/chrombpnet/wiki/FAQ\">FAQ</a> section in wiki. \
	<br> \
	<br> \
	<b> What to do if you are unsure if a given CWM motif is resembling the match_0 logo for example? </b> <br> \
	Get marginal footprint on the match_0 motif logo (using the command <em>chrombpnet footprints</em> and make sure that the bias models footprint is closer to that of controls with no motif \
	inserted - for examples look at <a href=\"https://github.com/kundajelab/chrombpnet/wiki/FAQ\">FAQ</a> )"

	table_profile = open(os.path.join(prefix,"evaluation/modisco_profile/motifs.html")).read().replace("./","./modisco_profile/").replace("width=\"240\"","width=\"240\", class=\"cover\"").replace(">pos_patterns.pattern",">pos_").replace(">neg_patterns.pattern",">neg_").replace("modisco_cwm_fwd","cwm_fwd").replace("modisco_cwm_rev","cwm_rev").replace("num_seqlets","NumSeqs").replace("dataframe","new")
	table_counts = open(os.path.join(prefix,"evaluation/modisco_counts/motifs.html")).read().replace("./","./modisco_counts/").replace("width=\"240\"","width=\"240\", class=\"cover\"").replace(">pos_patterns.pattern",">pos_").replace(">neg_patterns.pattern",">neg_").replace("modisco_cwm_fwd","cwm_fwd").replace("modisco_cwm_rev","cwm_rev").replace("num_seqlets","NumSeqs").replace("dataframe","new")

	# q-values with MEME's tomtom, p-values with --tomtom-lite; the q-value
	# wording is unchanged.
	stat_profile = modisco_table.match_stat(table_profile)
	stat_counts = modisco_table.match_stat(table_counts)
	tf_text_profile = tf_text_profile.replace(
		"qvals (qval0,qval1,qval2)", modisco_table.stat_description(stat_profile)).replace(
		"The qvals should", "The {} should".format(modisco_table.stat_name(stat_profile)))
	tf_text_counts = tf_text_counts.replace(
		"The qvals should", "The {} should".format(modisco_table.stat_name(stat_counts)))

	table_profile = remove_negs(table_profile)
	table_counts = remove_negs(table_counts)

	# 2. Combine them together using a long f-string

	html = f'''
			<body style="font-size:20px;">
				<h3>{bias_model_perf_hed}</h3>
				<p>{bias_model_perf_text}</p>
			</body>
				<body>
					{df.to_html()}
				</body>
				</body>
					{df1.to_html()}
				<body>
			<body style="font-size:20px;">
				<h3>{tf_hed}</h3>
				<p>{tf_text_profile}</p>
			</body>
			<body>
				 {table_profile}
			</body>
			<body style="font-size:20px;">
				<p>{tf_text_counts}</p>
			</body>
			<body>
				 {table_counts}
			</body>
		'''
			
	return html
	
def main(args):

	if args.file_prefix:
		fpx = args.file_prefix+"_"
	else:
		fpx = ""
		
	#prefix = "/home/anusri/full_run_tes/bias_model/"
	prefix = args.input_dir
	pd.set_option('colheader_justify', 'center')   # FOR TABLE <th>


	if (args.command=="pipeline"):
		page_title_text='Bias model training and quality check report'
		main_html = report_shell.page_shell(page_title_text)
		main_html += train_report(fpx,prefix)
		main_html += qc_report(fpx,prefix)

	if (args.command=="train"):
		page_title_text='Bias model training report'
		main_html = report_shell.page_shell(page_title_text)
		main_html += train_report(fpx,prefix)

	if (args.command=="qc"):
		page_title_text='Bias model quality check report'
		main_html = report_shell.page_shell(page_title_text)
		main_html += qc_report(fpx,prefix)
	
	
	report_shell.write_report(prefix, fpx, main_html, args.html_prefix)

if __name__=="__main__":
	args=read_args()
	main(args)
