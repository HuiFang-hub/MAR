import matplotlib
matplotlib.use('Agg') 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from matplotlib.ticker import MaxNLocator
import matplotlib.font_manager as fm
def plot_df(df,args,type ):
    # 设置图形大小
    # plt.figure(figsize=(10, 6))

    # 循环绘制曲线
    for column in df.columns[1:]:
        plt.plot(df[args.independent_v], df[column], label=column)
    # 添加标题和标签
    # plt.title('Prediction of DEH value')
    plt.xlabel(f'{args.independent_v}')
    plt.ylabel('DEH value')
    plt.legend()
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    # 显示图形
    # plt.savefig(f'{args.res_path}/Curve_{args.model}_{type}.pdf')
    plt.savefig(f'{args.res_curve_path}/Curve_{args.model}_{type}.png')
    plt.clf()
    plt.close()
    
    
# def pie(type ,ratio,labels,args):
#     colors = ['#8F7189','#ED7B61','#29A15C','#529AC9','#D5B2BD',
#               '#F6B09C','#9CCBA7','#99BADF','#866AA3','#C1B7CF',
#               '#83A0BE','#E18791']
    
#     pie_colors = [colors[i % len(colors)] for i in range(len(ratio))]
    
#     plt.pie(ratio, labels=labels, colors=pie_colors, autopct='%1.1f%%')
#     # plt.title('Feature Importance')
#     plt.tight_layout()
#     # plt.savefig(f'{args.res_importance_path}/importance_{args.model}_{type}.pdf')
#     plt.savefig(f'{args.res_importance_path}/importance_{args.model}_{type}.png')
#     plt.clf() 
#     plt.close()
    
def pie(type, ratio, labels, args):
    colors = ['#8F7189','#ED7B61','#29A15C','#529AC9','#D5B2BD',
              '#F6B09C','#9CCBA7','#99BADF','#866AA3','#C1B7CF',
              '#83A0BE','#E18791']
    
    pie_colors = [colors[i % len(colors)] for i in range(len(ratio))]
    
    # Do not display labels inside the pie chart
    wedges, _, autotexts = plt.pie(ratio, labels=None, colors=pie_colors, autopct='%1.1f%%',
                                   startangle=140, pctdistance=0.85)
    
    for autotext in autotexts:
        autotext.set_fontsize(9)  # Adjust the font size of the percentage text if needed
    
    # Display the legend outside the pie chart
    plt.legend(wedges, labels, loc="best", bbox_to_anchor=(1, 0.5), fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{args.res_importance_path}/importance_{args.model}_{type}.png', bbox_inches='tight')
    plt.clf() 
    plt.close()
    
def bar(type ,ratio,labels,args):
    plt.bar(height= ratio, x=labels)
    # plt.title('Feature Importance')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    # plt.savefig(f'{args.res_path}/importance_{args.model}_{type}.pdf')
    plt.savefig(f'{args.res_importance_path}/importance_{args.model}_{type}.png')
    plt.clf() 
    plt.close()
    
def heat_map(corr,fig_path):
    # font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'
    # font_properties = fm.FontProperties(fname=font_path, size=21)
    plt.figure(figsize=(20, 18))
    plt.rc('font', family='DejaVu Serif', size=26)  # 28, 32
    mask = ~np.tri(corr.shape[0], k=0, dtype=bool)  #Generates a mask of the lower triangle
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", mask=mask)
    # sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    # plt.title('The correlation of variables')
    plt.tight_layout()
    plt.xticks(rotation=30, ha='right') 
    # print(fig_path)
    plt.savefig(fig_path)
    # plt.savefig(f'{args.res_path}/{method}_Correlations.pdf')
    plt.clf() 
    plt.close()
    
def plot_scatter(Preds, Trues, args, fig_path=None):
    # 转换为NumPy数组以确保兼容性
    Preds = np.array(Preds)
    Trues = np.array(Trues)
    
    # 绘制散点图
    plt.scatter(Preds, Trues, s=10)
    
    # 绘制对角线 y = x
    min_val = min(min(Preds), min(Trues))
    max_val = max(max(Preds), max(Trues))
    plt.plot([min_val, max_val], [min_val, max_val], color='gray', linestyle='--')
    
    # 计算误差
    Deviation = Trues - Preds
    
    # 找到最大的绝对偏差值
    max_deviation = Deviation.max()
    min_deviation = Deviation.min()
    
    # 生成预测值范围
    pred_range = np.linspace(min_val, max_val, 100)
    
    # 计算误差带的上限和下限，以对角线 y=x 为中心
    upper_bound = pred_range + max_deviation
    lower_bound = pred_range + min_deviation
    

    plt.plot(pred_range, upper_bound, color='orange', linestyle='--')
    plt.plot(pred_range, lower_bound, color='blue', linestyle='--')
    plt.fill_between(pred_range, lower_bound, upper_bound, color='orange', alpha=0.2)
    
    # 添加标题和标签
    plt.title('Scatter Plot')
    plt.xlabel('Preds')
    plt.ylabel('Trues')
    # 添加图例
    plt.legend()
    
    # 调整布局
    plt.tight_layout()
    
    # 保存或显示图形
    if fig_path:
        plt.savefig(fig_path)
    else:
        plt.savefig(f'{args.res_value_scatter_path}/{args.model}_scatter_{args.seed}.png')
    
    # 清空并关闭图形
    plt.clf()
    plt.close()
    
def plot_violinplot(fig_path, df, x, y, order=None):
    color = ['#FF5575', '#FFD36A', '#6299FF', '#29B4B6']
    # color = ['#8F7289', '#ED7B61', '#29A15C','#529AC9']
    
    # Specify the path to the Times New Roman .ttf file
    font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'
    font_properties = fm.FontProperties(fname=font_path, size=21)
    
    # Adjust color order based on the 'order' argument
    if order:
        palette = {order[i]: color[i] for i in range(len(order))}
        df = df[df[x].isin(order)]
    else:
        palette = color

    # Plot violin plot with the specified palette
    ax = sns.violinplot(x=x, y=y, data=df, order=order, palette=palette)
    ax.set_xlabel('')
    ax.set_ylabel('')
    plt.xticks(fontsize=21, fontproperties=font_properties)
    plt.yticks(fontsize=21, fontproperties=font_properties)
    plt.tight_layout()

    # Save the plot
    plt.savefig(f'{fig_path}/{y}.png')
    plt.clf()
    plt.close()
    
    
def pairwise(df, png_path, pdf_path=None, group=False):
    font_size = 24
    font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'
    font_properties = fm.FontProperties(fname=font_path, size=font_size)
    # plt.figure(figsize=(20, 12))
    # font_size = 16
    sns.set(style="ticks")
    
    # 设置全局字体属性
    # plt.rc('font', family='DejaVu Serif', size=font_size)
    # plt.rcParams['font.family'] = 'serif'
    # plt.rcParams['font.serif'] = ['Times New Roman']
    
    if group:
        pair_plot = sns.pairplot(df, hue='class', palette='Set1', markers=["o", "s"])
    else:
        pair_plot = sns.pairplot(df)

    # 设置坐标轴刻度字体大小
    plt.xticks(fontsize=font_size,fontproperties=font_properties)
    plt.yticks(fontsize=font_size,fontproperties=font_properties)

    # 设置坐标轴标签字体大小
    for ax in pair_plot.axes.flatten():
        # ax.tick_params(axis='x', labelrotation=45)  # 旋转 x 轴刻度标签
        # ax.tick_params(axis='y', labelrotation=45)  # 旋转 y 轴刻度标签
        ax.set_xticklabels(ax.get_xticks().astype(int),fontsize=font_size, fontproperties=font_properties)
        ax.set_yticklabels(ax.get_yticks().astype(int),fontsize=font_size, fontproperties=font_properties)
        # ax.xaxis.label.set_rotation(15)  # 旋转 xlabel
        # ax.yaxis.label.set_rotation(15)
        ax.set_xlabel(ax.get_xlabel(), fontsize=26,fontproperties=font_properties)
        ax.set_ylabel(ax.get_ylabel(), fontsize=26,fontproperties=font_properties)
        
        
    pair_plot._legend.remove()
    
    # 确保图例不会遮挡内容
    # plt.setp(legend.get_texts(), fontsize=font_size)
    
    plt.tight_layout()
    plt.savefig(png_path)
    if pdf_path:
        plt.savefig(pdf_path)
    plt.clf()
    plt.close()
    
    
        # 修改图例位置
    # pair_plot.add_legend(title='Class', loc='upper right', fontsize=font_size)
     # 修改图例位置，平铺在最上方中央
   # 添加图例，设置位置为上方中央
    # legend = plt.legend(title='Class', loc='upper center', bbox_to_anchor=(0.5, 1.05), 
    #                     ncol=3, fontsize=font_size, frameon=False)
    # 修改图例位置
    # if group:
    #     pair_plot._legend.set_bbox_to_anchor((0.5, 0.95))
    #     pair_plot._legend.set_frame_on(True)



