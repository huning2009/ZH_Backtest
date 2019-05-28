# ZH_Backtest_Framework
搭建回测框架

2019-05-28 
首次创建
# API：
  market_neutral_backtest(factor, price):
    '''
    INPUT:
        weight：Muti-index df
        price：Muti-index df
        
    OUTPUT:
        performance summary: df
        累计收益曲线 + 最大回撤区间: plot
        
    '''
