"""Render the complete formal source intervals as a standalone scientific figure."""
import argparse,json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--font',type=Path,default=Path('/System/Library/Fonts/STHeiti Light.ttc'))
    args=parser.parse_args()
    report=json.loads(args.comparison.read_text())
    methods=['w1','fusion_long','diffusion_full_w64']
    if report['status']!='complete' or report['requests']!=270 or set(report['methods'])!=set(methods):
        raise ValueError('Expected the complete three-method natural comparison')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font=FontProperties(fname=str(args.font))
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    figure,axes=plt.subplots(2,3,figsize=(11.5,5.8),sharey=True)
    metrics=[('feature_conditional_acceptance','首 Feature 条件接受',100),
        ('pair_joint_acceptance','首 P/F 联合接受',100),
        ('mean_emitted_per_round','每轮推进',1)]
    colors=['#6f7986','#a07349','#287bb3']
    labels=['W1','fusion_long','DFlash Full 64']
    points=[]
    for row,(cohort,title,expected_requests) in enumerate([('bos','BOS · 30 来源',30),('completion','Completion · 30 来源 × 2 seeds',60)]):
        for column,(metric,heading,scale) in enumerate(metrics):
            axis=axes[row,column]
            for index,method in enumerate(methods):
                own=report['cohorts'][cohort][method]
                if own['requests']!=expected_requests or own['sources']!=30:raise ValueError('Formal source coverage differs')
                interval=own['source_mean_intervals'][metric]
                value=interval['mean']*scale;lower,upper=[x*scale for x in interval['ci95']]
                axis.errorbar(value,index,xerr=[[value-lower],[upper-value]],fmt='o',markersize=6,color=colors[index],capsize=3,lw=1.5)
                points.append(dict(cohort=cohort,method=method,metric=metric,mean=interval['mean'],ci95=interval['ci95'],
                    sources=interval['sources'],input_sources=own['sources'],requests=expected_requests,
                    undefined_requests=interval.get('undefined_requests',0)))
            axis.set_yticks(range(3),labels)
            axis.set_ylim(2.65,-.6)
            axis.grid(axis='x',alpha=.18)
            axis.set_xlabel('接受率 (%)' if scale==100 else 'token / round',fontproperties=font)
            if column==0:
                axis.set_ylabel(title,fontproperties=font)
                axis.axvline(40,color='#989da4',ls=':',lw=1)
            if row==0:axis.set_title(heading,fontproperties=font)
    figure.suptitle('正式自然生成：接受与推进',fontproperties=font,fontsize=16,y=.97)
    figure.text(.06,.025,'点为可评价来源等权均值，线为 95% 来源区间。Completion 两个 seed 保留在同一来源；DFlash 为集合接受。未定义请求单列。',fontproperties=font,fontsize=9)
    figure.tight_layout(rect=(0,.065,1,.94))
    args.out.mkdir(parents=True,exist_ok=True)
    figure.savefig(args.out/'formal_acceptance.png',dpi=200)
    figure.savefig(args.out/'formal_acceptance.pdf')
    plt.close(figure)
    (args.out/'figure_sources.json').write_text(json.dumps(dict(comparison=str(args.comparison),points=points,
        scope='Prespecified natural BOS30 and Completion30x2; 5000 paired source resamples',
        feature_scope='First Feature conditional candidate-set acceptance; candidate width differs between single and multi-candidate methods'),indent=2)+'\n')
    print('Formal figure saved',args.out)


if __name__=='__main__':main()
