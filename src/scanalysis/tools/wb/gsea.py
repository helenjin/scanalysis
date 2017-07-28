import re
import os
import shutil
import shlex
import glob
import pandas as pd

from subprocess import call, Popen, PIPE

def _gmt_options():
    mouse_options = os.listdir(os.path.expanduser('~/.scanalysis/tools/mouse'))
    human_options = os.listdir(os.path.expanduser('~/.scanalysis/tools/human'))
    print('Available GSEA .gmt files:\n\nmouse:\n{m}\n\nhuman:\n{h}\n'.format(
            m='\n'.join(mouse_options),
            h='\n'.join(human_options)))
    print('Please specify the gmt_file parameter as gmt_file=(organism, filename)')


def _gsea_process(c, diffusion_map_correlations, output_stem, gmt_file):

    # save the .rnk file
    out_dir, out_prefix = os.path.split(output_stem)
    genes_file = '{stem}_cmpnt_{component}.rnk'.format(
            stem=output_stem, component=c)
    ranked_genes = diffusion_map_correlations.ix[:, c]\
        .sort_values(inplace=False, ascending=False)

    # set any NaN to 0
    ranked_genes = ranked_genes.fillna(0)

    # dump to file
    pd.DataFrame(ranked_genes).to_csv(genes_file, sep='\t', header=False)

    # Construct the GSEA call
    cmd = shlex.split(
        'java -cp {user}/.wishbone/tools/gsea2-2.2.1.jar -Xmx1g '
        'xtools.gsea.GseaPreranked -collapse false -mode Max_probe -norm meandiv '
        '-nperm 1000 -include_only_symbols true -make_sets true -plot_top_x 0 '
        '-set_max 500 -set_min 50 -zip_report false -gui false -rnk {rnk} '
        '-rpt_label {out_prefix}_{component} -out {out_dir}/ -gmx {gmt_file}'
        ''.format(user=os.path.expanduser('~'), rnk=genes_file,
                  out_prefix=out_prefix, component=c, out_dir=out_dir,
                  gmt_file=gmt_file))

    # Call GSEA
    p = Popen(cmd, stderr=PIPE)
    _, err = p.communicate()

    # remove annoying suffix from GSEA
    if err:
        return err
    else:
        pattern = out_prefix + '_' + str(c) + '.GseaPreranked.[0-9]*'
        repl = out_prefix + '_' + str(c)
        files = os.listdir(out_dir)
        for f in files:
            mo = re.match(pattern, f)
            if mo:
                curr_name = mo.group(0)
                shutil.move('{}/{}'.format(out_dir, curr_name),
                            '{}/{}'.format(out_dir, repl))
                return err

        # execute if file cannot be found
        return b'GSEA output pattern was not found, and could not be changed.'

def run_gsea(diffusion_map_correlations, output_stem, gmt_file=None,
    components=None, enrichment_threshold=1e-1):
    """ Run GSEA using gene rankings from diffusion map correlations
    :param output_stem: the file location and prefix for the output of GSEA
    :param gmt_file: GMT file containing the gene sets. Use None to see a list of options
    :param components: Iterable of integer component numbers
    :param enrichment_threshold: FDR corrected p-value significance threshold for gene set enrichments
    :return: Dictionary containing the top enrichments for each component
    
    *Please run run_diffusion_map_correlations() before running GSEA to annotate those components.
    """

    out_dir, out_prefix = os.path.split(output_stem)
    out_dir += '/'
    os.makedirs(out_dir, exist_ok=True)

    if not gmt_file:
        _gmt_options()
        return
    else:
        if not len(gmt_file) == 2:
            raise ValueError('gmt_file should be a tuple of (organism, filename).')
        gmt_file = os.path.expanduser('~/.wishbone/tools/{}/{}').format(*gmt_file)

    if components is None:
        components = diffusion_map_correlations.columns

    # Run GSEA
    print('If running in notebook, please look at the command line window for GSEA progress log')
    reports = dict()
    for c in components:
        res = _gsea_process( c, diffusion_map_correlations,
            output_stem, gmt_file )
        # Load results
        if res == b'':
            # Positive correlations
            df = pd.DataFrame.from_csv(glob.glob(output_stem + '_%d/gsea*pos*xls' % c)[0], sep='\t')
            reports[c] = dict()
            reports[c]['pos'] = df['FDR q-val'][0:5]
            reports[c]['pos'] = reports[c]['pos'][reports[c]['pos'] < enrichment_threshold]

            # Negative correlations
            df = pd.DataFrame.from_csv(glob.glob(output_stem + '_%d/gsea*neg*xls' % c)[0], sep='\t')
            reports[c]['neg'] = df['FDR q-val'][0:5]
            reports[c]['neg'] = reports[c]['neg'][reports[c]['neg'] < enrichment_threshold]

    # Return results
    return reports
