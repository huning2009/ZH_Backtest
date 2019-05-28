# ZH_Backtest
搭建回测框架

# 更新日志：
- 2019-05-28：首次创建


## 函数式API
### market_neutral_backtest

```python
market_neutral_backtest(factor, price)
```

市场中性回测

__参数__
- factor_data : pd.DataFrame - MultiIndex. A MultiIndex DataFrame indexed by date (level 0) and asset (level 1)
- prices : pd.DataFrame - MultiIndex. A MultiIndex DataFrame indexed by date (level 0) and asset (level 1)

__输出__
- performance summary: pd.DataFrame
- 累计收益曲线 + 最大回撤区间: plot

----
