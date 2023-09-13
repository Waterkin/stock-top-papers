# About

The sole purpose of this repository is to help me organize recent academic papers *with codes* related to _stock price prediction_, _quantitative trading_, _risk modeling_. This is a non-exhausting list, even though I'll try to keep it updated...
Feel free to suggest decent papers via a PR.
If you find this repository helpful consider leaving a :star:

# Table of Contents

* [Sorted by Time](#sorted-by-time)
    * [2023](#2023)
    * [2022](#2022)
    * [2021](#2021)
    * [2020](#2020)
    * [Older Papers](#older-papers)
* [Sorted by Tasks](#sorted-by-tasks)
    * [Stock Price Prediction](#stock-price-prediction)
    * [Stock Trading](#stock-trading)
    * [Asset Pricing](#asset-pricing)
    * [Risk Modeling](#risk-modeling)
* [Sorted by Models](#sorted-by-models)
    * [Diffusion Model](#diffusion-model)
    * [Transformer](#transformer)
    * [Variational Autoencoder](#variational-autoencoder)
* [Sorted by Methods](#sorted-by-methods)
    * [NLP-based Methods](#nlp-based-methods)
    * [Graph Learning](#graph-learning)
    * [Reinforcement-learning](#reinforcement-learning)
    * [Multi-task Learning](#multi-task-learning)
    * [Meta-learning](#meta-learning)
    * [Contrastive Learning](#contrastive-learning)
    * [Curriculum Learning](#curriculum-learning)
    * [Ensemble Learning](#ensemble-learning)

# Sorted by Time

[Back to top](#table-of-contents)

## 2023

* (AAAI 2023) StockEmotions: Discover Investor Emotions for Financial Sentiment Analysis and Multivariate Time Series \[[Paper](https://www.semanticscholar.org/paper/StockEmotions%3A-Discover-Investor-Emotions-for-and-Lee-Youn/13e4e72303a630c4b7e581d59facfc014c846a24)\]\[[Code](https://github.com/adlnlp/StockEmotions)\]
* (ICDE 2023) Relational Temporal Graph Convolutional Network for Ranking-based Stock Prediction \[[Paper](https://ieeexplore.ieee.org/document/10184655)\]\[[Code](https://github.com/zhengzetao/RTGCN)\]
* (IJCAI 2023) StockFormer: Learning Hybrid Trading Machines with Predictive Coding \[[Paper](https://www.ijcai.org/proceedings/2023/0530.pdf)\]\[[Code](https://github.com/gsyyysg/StockFormer)\]
* (TKDE 2023) Stock Movement Prediction Based on Bi-typed and Hybrid-relational Market Knowledge Graph via Dual Attention Networks \[[Paper](https://www.semanticscholar.org/paper/Stock-Movement-Prediction-Based-on-Bi-Typed-Market-Zhao-Du/dbaf9ff32a00161d777f6f5cd50e4028d733bd0d)\]\[[Code](https://github.com/trytodoit227/DANSMP)\]
* (KDD 2023) DoubleAdapt: A Meta-learning Approach to Incremental Learning for Stock Trend Forecasting \[[Paper](https://dl.acm.org/doi/pdf/10.1145/3580305.3599315)\]\[[Code](https://github.com/SJTU-Quant/qlib)\]
* (KDD 2023) Efficient Continuous Space Policy Optimization for High-frequency Trading \[[Paper](https://dl.acm.org/doi/pdf/10.1145/3580305.3599813)\]\[[Code](https://github.com/finint/DRPO)\]
* (KDD 2023)Generating Synergistic Formulaic Alpha Collections via Reinforcement Learning \[[Paper](https://arxiv.org/pdf/2306.12964)\]\[[Code](https://github.com/RL-MLDM/alphagen)\]
* (2023) FinGPT: Democratizing Internet-scale Data for Financial Large Language Models \[[Paper](https://arxiv.org/abs/2307.10485)\]\[[Code](https://github.com/AI4Finance-Foundation/FinGPT)\]
* (2023) Generative Meta-Learning Robust Quality-Diversity Portfolio \[[Paper](https://arxiv.org/abs/2307.07811)\]\[[Code](https://github.com/kayuksel/generative-opt)\]
* (2023) MOPO-LSI: A User Guide \[[Paper](https://arxiv.org/abs/2307.01719)\]\[[Code](https://github.com/irecsys/MOPO-LSI)\]
* (2023) Diffusion Variational Autoencoder for Tackling Stochasticity in Multi-Step Regression Stock Price Prediction \[[Paper](https://arxiv.org/abs/2309.00073)\]\[[Code](https://github.com/koa-fin/dva)\]

## 2022

* (CIKM 2022) Temporal and Heterogeneous Graph Neural Network for Financial Time Series Prediction \[[Paper](https://dl.acm.org/doi/abs/10.1145/3511808.3557089)\]\[[Code](https://github.com/finint/THGNN)\]
* (CIKM 2022) DeepScalper: A Risk-Aware Reinforcement Learning Framework to Capture Fleeting Intraday Trading Opportunities \[[Paper](https://www.semanticscholar.org/paper/DeepScalper%3A-A-Risk-Aware-Reinforcement-Learning-to-Sun-He/d57743b30ca50b1480a72ab41a0564f20f183e92)\]\[[Code](https://github.com/TradeMaster-NTU/TradeMaster)\]
* (ICASSP 2022) Hypergraph-Based Reinforcement Learning for Stock Portfolio Selection \[[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9747138)\]\[[Code](https://github.com/Linlinmm/Stock-Portfolio-Management)\]
* (AAAI 2022) FactorVAE: A Probabilistic Dynamic Factor Model Based on Variational Autoencoder for Predicting Cross-Sectional Stock Returns \[[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/20369/20128)\]\[[Code](https://github.com/UePG-21/facvae)\]
* (NeurIPS workshop 2022) FinRL-Meta: Market Environments and Benchmarks for Data-Driven Financial Reinforcement Learning \[[Paper](https://openreview.net/pdf?id=LkAFwrqdRY6)\]\[[Code](https://github.com/AI4Finance-Foundation/FinRL-Meta)\]
* (EMNLP workshop 2022) Astock: A New Dataset and Automated Stock Trading based on Stock-specific News Analyzing Model \[[Paper](https://aclanthology.org/2022.finnlp-1.24/)\]\[[Code](https://github.com/JinanZou/Astock)\]
* (2022) Trade When Opportunity Comes: Price Movement Forecasting via Locality-Aware Attention and Adaptive Refined Labeling \[[Paper](https://arxiv.org/pdf/2107.11972.pdf)\]\[[Code](https://tinyurl.com/LARA-KDD2022)\]
* (2022) Inductive Representation Learning on Dynamic Stock Co-Movement Graphs for Stock Predictions \[[Paper](https://www.biz.uiowa.edu/faculty/kangzhao/pub/ijoc_2022.pdf)\]\[[Code](https://github.com/Hugo-CAS/Dynamic-Stock-Co-Movement-Graphs-for-Stock-Predictions)\]
* (2022) Quantitative Stock Investment by Routing Uncertainty-Aware Trading Experts: A Multi-Task Learning Approach \[[Paper](https://arxiv.org/abs/2207.07578)\]\[[Code](https://github.com/johnson7788/AlphaMix)\]
* (2022) Safe-FinRL: A Low Bias and Variance Deep Reinforcement Learning Implementation for High-Freq Stock Trading \[[Paper](https://arxiv.org/abs/2206.05910)\]\[[Code](https://github.com/Tsedao/Safe-FinRL)\]

## 2021

* (TKDE 2021) FinGAT: Financial Graph Attention Networks for Recommending Top-K Profitable Stocks \[[Paper](https://arxiv.org/abs/2106.10159)\]\[[Code](https://github.com/Roytsai27/Financial-GraphAttention)\]
* (TKDD 2021) Graph-Based Stock Recommendation by Time-Aware Relational Attention Network \[[Paper](https://dl.acm.org/doi/10.1145/3451397)\]\[[Code](https://github.com/xiaoting135/TRAN)\]
* (CIKM 2021) Attention Based Dynamic Graph Learning Framework for Asset Pricing \[[Paper](https://dl.acm.org/doi/abs/10.1145/3459637.3482413)\]\[[Code](https://github.com/Ajim63/Attention-Based-Dynamic-Graph-Learning-Framework-for-Asset-Pricing)\]
* (CIKM 2021) Stock Trend Prediction with Multi-granularity Data: A Contrastive Learning Approach with Adaptive Fusion \[[Paper](http://staff.ustc.edu.cn/~cheneh/paper_pdf/2021/Min-Hou-CIKM.pdf)\]\[[Code](https://github.com/CMLF-git-dev/CMLF)\]
* (KDD 2021) Accurate Multivariate Stock Movement Prediction via Data-Axis Transformer with Multi-Level Contexts \[[Paper](https://datalab.snu.ac.kr/~ukang/papers/dtmlKDD21.pdf)\]\[[Code](https://github.com/simonjisu/DTML-pytorch)\]
* (EACL 2021) FAST: Financial News and Tweet Based Time Aware Network for Stock Trading \[[Paper](https://aclanthology.org/2021.eacl-main.185)\]\[[Code](https://github.com/midas-research/fast-eacl)\]
* (NAACL 2021) Quantitative Day Trading from Natural Language using Reinforcement Learning \[[Paper](https://aclanthology.org/2021.naacl-main.316)\]\[[Code](https://github.com/midas-research/profit-naacl)\]
* (AAAI 2021) Stock Selection via Spatiotemporal Hypergraph Attention Network: A Learning to Rank Approach \[[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/16127)\]\[[Code](https://github.com/midas-research/sthan-sr-aaai)\]
* (AAAI 2021) Universal Trading for Order Execution with Oracle Policy Distillation \[[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/16083)\]\[[Code](https://github.com/TradeMaster-NTU/TradeMaster)\]
* (AAAI 2021) DeepTrader: A Deep Reinforcement Learning Approach for Risk-Return Balanced Portfolio Management with Market Conditions Embedding \[[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/16144)\]\[[Code](https://github.com/TradeMaster-NTU/TradeMaster)\]
* (UAI 2021) Modeling Financial Uncertainty with Multivariate Temporal Entropy-based Curriculums \[[Paper](https://www.auai.org/uai2021/pdf/uai2021.638.preliminary.pdf)\]\[[Code](https://github.com/midas-research/finclass-uai)\]
* (WWW 2021) Exploring the Scale-Free Nature of Stock Markets: Hyperbolic Graph Learning for Algorithmic Trading \[[Paper](https://dl.acm.org/doi/abs/10.1145/3442381.3450095)\]\[[Code](https://github.com/midas-research/hyper-stockgat-www)\]
* (SIGIR 2021) Hyperbolic Online Time Stream Modeling \[[Paper](https://dl.acm.org/doi/10.1145/3404835.3463119)\]\[[Code](https://github.com/midas-research/hyperbolic-tlstm-sigir)\]
* (ICAIF 2021) FinRL: deep reinforcement learning framework to automate trading in quantitative finance \[[Paper](https://dl.acm.org/doi/10.1145/3490354.3494366)\]\[[Code](https://github.com/AI4Finance-Foundation/FinRL)\]
* (ACL Findings 2021) Trade the Event: Corporate Events Detection for News-Based Event-Driven Trading \[[Paper](https://aclanthology.org/2021.findings-acl.186.pdf)\]\[[Code](https://github.com/Zhihan1996/TradeTheEvent)\]
* (2021) HIST: A Graph-based Framework for Stock Trend Forecasting via Mining Concept-Oriented Shared Information \[[Paper](http://arxiv.org/abs/2110.13716)\]\[[Code](https://github.com/Wentao-Xu/HIST)\]

## 2020

* (AAAI 2020) Reinforcement-Learning based Portfolio Management with Augmented Asset Movement Prediction States \[[Paper](https://arxiv.org/abs/2002.05780)\]\[[Code](https://github.com/TradeMaster-NTU/TradeMaster)\]
* (EMNLP 2020) Deep Attentive Learning for Stock Movement Prediction From Social Media Text and Company Correlations \[[Paper](https://www.aclweb.org/anthology/2020.emnlp-main.676)\]\[[Code](https://github.com/midas-research/man-sf-emnlp)\]
* (ICDM 2020) Spatiotemporal Hypergraph Convolution Network for Stock Movement Forecasting \[[Paper](https://ieeexplore.ieee.org/document/9338303/)\]\[[Code](https://github.com/midas-research/sthgcn-icdm)\]
* (ICDM 2020) DoubleEnsemble: A New Ensemble Method Based on Sample Reweighting and Feature Selection for Financial Data Analysis \[[Paper](https://www.semanticscholar.org/paper/DoubleEnsemble%3A-A-New-Ensemble-Method-Based-on-and-Zhang-Li/96d8383288eba50d69f516522154cf52625c7a4f)\]\[[Code](https://github.com/Sakura-Fire-Capital/DoubleEnsembleML)\]
* (ACM MM 2020) Multimodal Multi-Task Financial Risk Forecasting \[[Paper](https://dl.acm.org/doi/10.1145/3394171.3413752)\]\[[Code](https://github.com/midas-research/multimodal-financial-forecasting)\]
* (IJCAI 2020) F-HMTC: Detecting Financial Events for Investment Decisions Based on Neural Hierarchical Multi-Label Text Classification \[[Paper](https://www.ijcai.org/proceedings/2020/0619.pdf)\]\[[Code](https://github.com/finint/F-HMTC)\]
* (IJCAI 2020) An End-to-End Optimal Trade Execution Framework based on Proximal Policy Optimization \[[Paper](https://www.ijcai.org/Proceedings/2020/627?msclkid=a2b6ad5db7ca11ecb537627a9ca1d4f6)\]\[[Code](https://github.com/TradeMaster-NTU/TradeMaster)\]
* (CIKM 2020) Fusing Global Domain Information and Local Semantic Information to Classify Financial Documents \[[Paper](https://dl.acm.org/doi/10.1145/3340531.3412707)\]\[[Code](https://github.com/finint/graphSEAT)\]
* (ESWA 2020) Time-driven feature-aware jointly deep reinforcement learning for financial signal representation and algorithmic trading \[[Paper](https://www.sciencedirect.com/science/article/pii/S0957417419305822?via%3Dihub)\]\[[Code](https://github.com/Lingfeng158/TFJ-DRL-Replication)\]
* (2020) Open Source Cross-Sectional Asset Pricing \[[Paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3604626)\]\[[Code](https://github.com/Sakura-Fire-Capital/CrossSection)\]

## Older Papers

* (IJCAI 2019) Enhancing Stock Movement Prediction with Adversarial Training \[[Paper](https://www.ijcai.org/proceedings/2019/0810.pdf)\]\[[Code](https://github.com/fulifeng/Adv-ALSTM)\]
* (TIS 2019) Temporal Relational Ranking for Stock Prediction \[[Paper](https://arxiv.org/pdf/1809.09441.pdf)\]\[[Code](https://github.com/fulifeng/Temporal_Relational_Stock_Ranking)\]
* (2019) HATS: A Hierarchical Graph Attention Network for Stock Movement Prediction \[[Paper](https://arxiv.org/abs/1908.07999)\]\[[Code](https://github.com/dmis-lab/hats)\]
* (KDD 2018) Investor-Imitator: A Framework for Trading Knowledge Extraction \[[Paper](https://www.kdd.org/kdd2018/accepted-papers/view/investor-imitator-a-framework-for-trading-knowledge-extraction)\]\[[Code](https://github.com/TradeMaster-NTU/TradeMaster)\]
* (ACL 2018) Stock Movement Prediction from Tweets and Historical Prices \[[Paper](http://aclweb.org/anthology/P18-1183)\]\[[Code](https://github.com/yumoxu/stocknet-code)\]
* (WSDM 2018) Listening to Chaotic Whispers: A Deep Learning Framework for News-oriented Stock Trend Prediction \[[Paper](https://arxiv.org/abs/1712.02136)\]\[[Code](https://github.com/donghyeonk/han)\]
* (KDD 2017) Stock Price Prediction via Discovering Multi-Frequency Trading Patterns \[[Paper](https://dl.acm.org/doi/10.1145/3097983.3098117)\]\[[Code](https://github.com/microsoft/qlib)\]
* (2017) A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem \[[Paper](https://arxiv.org/abs/1706.10059)\]\[[Code](https://github.com/TradeMaster-NTU/TradeMaster)\]


# Sorted by Tasks

[Back to top](#table-of-contents)
## Stock Price Prediction

* (ICDE 2023) Relational Temporal Graph Convolutional Network for Ranking-based Stock Prediction \[[Paper](https://ieeexplore.ieee.org/document/10184655)\]\[[Code](https://github.com/zhengzetao/RTGCN)\]
* (TKDE 2023) Stock Movement Prediction Based on Bi-typed and Hybrid-relational Market Knowledge Graph via Dual Attention Networks \[[Paper](https://www.semanticscholar.org/paper/Stock-Movement-Prediction-Based-on-Bi-Typed-Market-Zhao-Du/dbaf9ff32a00161d777f6f5cd50e4028d733bd0d)\]\[[Code](https://github.com/trytodoit227/DANSMP)\]
* (KDD 2023) DoubleAdapt: A Meta-learning Approach to Incremental Learning for Stock Trend Forecasting \[[Paper](https://dl.acm.org/doi/pdf/10.1145/3580305.3599315)\]\[[Code](https://github.com/SJTU-Quant/qlib)\]
* (2023) Diffusion Variational Autoencoder for Tackling Stochasticity in Multi-Step Regression Stock Price Prediction \[[Paper](https://arxiv.org/abs/2309.00073)\]\[[Code](https://github.com/koa-fin/dva)\]
* (CIKM 2022) Temporal and Heterogeneous Graph Neural Network for Financial Time Series Prediction \[[Paper](https://dl.acm.org/doi/abs/10.1145/3511808.3557089)\]\[[Code](https://github.com/finint/THGNN)\]
* (2022) Inductive Representation Learning on Dynamic Stock Co-Movement Graphs for Stock Predictions \[[Paper](https://www.biz.uiowa.edu/faculty/kangzhao/pub/ijoc_2022.pdf)\]\[[Code](https://github.com/Hugo-CAS/Dynamic-Stock-Co-Movement-Graphs-for-Stock-Predictions)\]
* (CIKM 2021) Stock Trend Prediction with Multi-granularity Data: A Contrastive Learning Approach with Adaptive Fusion \[[Paper](http://staff.ustc.edu.cn/~cheneh/paper_pdf/2021/Min-Hou-CIKM.pdf)\]\[[Code](https://github.com/CMLF-git-dev/CMLF)\]
* (KDD 2021) Accurate Multivariate Stock Movement Prediction via Data-Axis Transformer with Multi-Level Contexts \[[Paper](https://datalab.snu.ac.kr/~ukang/papers/dtmlKDD21.pdf)\]\[[Code](https://github.com/simonjisu/DTML-pytorch)\]
* (2021) HIST: A Graph-based Framework for Stock Trend Forecasting via Mining Concept-Oriented Shared Information \[[Paper](http://arxiv.org/abs/2110.13716)\]\[[Code](https://github.com/Wentao-Xu/HIST)\]
* (EMNLP 2020) Deep Attentive Learning for Stock Movement Prediction From Social Media Text and Company Correlations \[[Paper](https://www.aclweb.org/anthology/2020.emnlp-main.676)\]\[[Code](https://github.com/midas-research/man-sf-emnlp)\]
* (ICDM 2020) Spatiotemporal Hypergraph Convolution Network for Stock Movement Forecasting \[[Paper](https://ieeexplore.ieee.org/document/9338303/)\]\[[Code](https://github.com/midas-research/sthgcn-icdm)\]

## Stock Trading

To be done...

## Asset Pricing

To be done...

## Risk Modeling

To be done...


# Sorted by Models

[Back to top](#table-of-contents)

## Diffusion Model

* (2023) Diffusion Variational Autoencoder for Tackling Stochasticity in Multi-Step Regression Stock Price Prediction \[[Paper](https://arxiv.org/abs/2309.00073)\]\[[Code](https://github.com/koa-fin/dva)\]

## Transformer

* (IJCAI 2023) StockFormer: Learning Hybrid Trading Machines with Predictive Coding \[[Paper](https://www.ijcai.org/proceedings/2023/0530.pdf)\]\[[Code](https://github.com/gsyyysg/StockFormer)\]
* (KDD 2021) Accurate Multivariate Stock Movement Prediction via Data-Axis Transformer with Multi-Level Contexts \[[Paper](https://datalab.snu.ac.kr/~ukang/papers/dtmlKDD21.pdf)\]\[[Code](https://github.com/simonjisu/DTML-pytorch)\]

## Variational Autoencoder

* (2023) Diffusion Variational Autoencoder for Tackling Stochasticity in Multi-Step Regression Stock Price Prediction \[[Paper](https://arxiv.org/abs/2309.00073)\]\[[Code](https://github.com/koa-fin/dva)\]
* (AAAI 2022) FactorVAE: A Probabilistic Dynamic Factor Model Based on Variational Autoencoder for Predicting Cross-Sectional Stock Returns \[[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/20369/20128)\]\[[Code](https://github.com/UePG-21/facvae)\]

# Sorted by Methods

[Back to top](#table-of-contents)

## NLP-based Methods

* (AAAI 2023) StockEmotions: Discover Investor Emotions for Financial Sentiment Analysis and Multivariate Time Series \[[Paper](https://www.semanticscholar.org/paper/StockEmotions%3A-Discover-Investor-Emotions-for-and-Lee-Youn/13e4e72303a630c4b7e581d59facfc014c846a24)\]\[[Code](https://github.com/adlnlp/StockEmotions)\]
* (TKDE 2023) Stock Movement Prediction Based on Bi-typed and Hybrid-relational Market Knowledge Graph via Dual Attention Networks \[[Paper](https://www.semanticscholar.org/paper/Stock-Movement-Prediction-Based-on-Bi-Typed-Market-Zhao-Du/dbaf9ff32a00161d777f6f5cd50e4028d733bd0d)\]\[[Code](https://github.com/trytodoit227/DANSMP)\]
* (2023) FinGPT: Democratizing Internet-scale Data for Financial Large Language Models \[[Paper](https://arxiv.org/abs/2307.10485)\]\[[Code](https://github.com/AI4Finance-Foundation/FinGPT)\]
* (EMNLP workshop 2022) Astock: A New Dataset and Automated Stock Trading based on Stock-specific News Analyzing Model \[[Paper](https://aclanthology.org/2022.finnlp-1.24/)\]\[[Code](https://github.com/JinanZou/Astock)\]
* (NAACL 2021) Quantitative Day Trading from Natural Language using Reinforcement Learning \[[Paper](https://aclanthology.org/2021.naacl-main.316)\]\[[Code](https://github.com/midas-research/profit-naacl)\]
* (ACL Findings 2021) Trade the Event: Corporate Events Detection for News-Based Event-Driven Trading \[[Paper](https://aclanthology.org/2021.findings-acl.186.pdf)\]\[[Code](https://github.com/Zhihan1996/TradeTheEvent)\]
* (EMNLP 2020) Deep Attentive Learning for Stock Movement Prediction From Social Media Text and Company Correlations \[[Paper](https://www.aclweb.org/anthology/2020.emnlp-main.676)\]\[[Code](https://github.com/midas-research/man-sf-emnlp)\]
* (IJCAI 2020) F-HMTC: Detecting Financial Events for Investment Decisions Based on Neural Hierarchical Multi-Label Text Classification \[[Paper](https://www.ijcai.org/proceedings/2020/0619.pdf)\]\[[Code](https://github.com/finint/F-HMTC)\]
* (CIKM 2020) Fusing Global Domain Information and Local Semantic Information to Classify Financial Documents \[[Paper](https://dl.acm.org/doi/10.1145/3340531.3412707)\]\[[Code](https://github.com/finint/graphSEAT)\]
* (ACL 2018) Stock Movement Prediction from Tweets and Historical Prices \[[Paper](http://aclweb.org/anthology/P18-1183)\]\[[Code](https://github.com/yumoxu/stocknet-code)\]

## Graph Learning

* (ICDE 2023) Relational Temporal Graph Convolutional Network for Ranking-based Stock Prediction \[[Paper](https://ieeexplore.ieee.org/document/10184655)\]\[[Code](https://github.com/zhengzetao/RTGCN)\]
* (TKDE 2023) Stock Movement Prediction Based on Bi-typed and Hybrid-relational Market Knowledge Graph via Dual Attention Networks \[[Paper](https://www.semanticscholar.org/paper/Stock-Movement-Prediction-Based-on-Bi-Typed-Market-Zhao-Du/dbaf9ff32a00161d777f6f5cd50e4028d733bd0d)\]\[[Code](https://github.com/trytodoit227/DANSMP)\]
* (CIKM 2022) Temporal and Heterogeneous Graph Neural Network for Financial Time Series Prediction \[[Paper](https://dl.acm.org/doi/abs/10.1145/3511808.3557089)\]\[[Code](https://github.com/finint/THGNN)\]
* (ICASSP 2022) Hypergraph-Based Reinforcement Learning for Stock Portfolio Selection \[[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9747138)\]\[[Code](https://github.com/Linlinmm/Stock-Portfolio-Management)\]
* (2022) Inductive Representation Learning on Dynamic Stock Co-Movement Graphs for Stock Predictions \[[Paper](https://www.biz.uiowa.edu/faculty/kangzhao/pub/ijoc_2022.pdf)\]\[[Code](https://github.com/Hugo-CAS/Dynamic-Stock-Co-Movement-Graphs-for-Stock-Predictions)\]
* (TKDE 2021) FinGAT: Financial Graph Attention Networks for Recommending Top-K Profitable Stocks \[[Paper](https://arxiv.org/abs/2106.10159)\]\[[Code](https://github.com/Roytsai27/Financial-GraphAttention)\]
* (TKDD 2021) Graph-Based Stock Recommendation by Time-Aware Relational Attention Network \[[Paper](https://dl.acm.org/doi/10.1145/3451397)\]\[[Code](https://github.com/xiaoting135/TRAN)\]
* (CIKM 2021) Attention Based Dynamic Graph Learning Framework for Asset Pricing \[[Paper](https://dl.acm.org/doi/abs/10.1145/3459637.3482413)\]\[[Code](https://github.com/Ajim63/Attention-Based-Dynamic-Graph-Learning-Framework-for-Asset-Pricing)\]
* (AAAI 2021) Stock Selection via Spatiotemporal Hypergraph Attention Network: A Learning to Rank Approach \[[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/16127)\]\[[Code](https://github.com/midas-research/sthan-sr-aaai)\]
* (WWW 2021) Exploring the Scale-Free Nature of Stock Markets: Hyperbolic Graph Learning for Algorithmic Trading \[[Paper](https://dl.acm.org/doi/abs/10.1145/3442381.3450095)\]\[[Code](https://github.com/midas-research/hyper-stockgat-www)\]
* (SIGIR 2021) Hyperbolic Online Time Stream Modeling \[[Paper](https://dl.acm.org/doi/10.1145/3404835.3463119)\]\[[Code](https://github.com/midas-research/hyperbolic-tlstm-sigir)\]
* (2021) HIST: A Graph-based Framework for Stock Trend Forecasting via Mining Concept-Oriented Shared Information \[[Paper](http://arxiv.org/abs/2110.13716)\]\[[Code](https://github.com/Wentao-Xu/HIST)\]
* (ICDM 2020) Spatiotemporal Hypergraph Convolution Network for Stock Movement Forecasting \[[Paper](https://ieeexplore.ieee.org/document/9338303/)\]\[[Code](https://github.com/midas-research/sthgcn-icdm)\]
* (2019) HATS: A Hierarchical Graph Attention Network for Stock Movement Prediction \[[Paper](https://arxiv.org/abs/1908.07999)\]\[[Code](https://github.com/dmis-lab/hats)\]



## Reinforcement-learning

* (KDD 2023) Efficient Continuous Space Policy Optimization for High-frequency Trading \[[Paper](https://dl.acm.org/doi/pdf/10.1145/3580305.3599813)\]\[[Code](https://github.com/finint/DRPO)\]
* (KDD 2023)Generating Synergistic Formulaic Alpha Collections via Reinforcement Learning \[[Paper](https://arxiv.org/pdf/2306.12964)\]\[[Code](https://github.com/RL-MLDM/alphagen)\]
* (CIKM 2022) DeepScalper: A Risk-Aware Reinforcement Learning Framework to Capture Fleeting Intraday Trading Opportunities \[[Paper](https://www.semanticscholar.org/paper/DeepScalper%3A-A-Risk-Aware-Reinforcement-Learning-to-Sun-He/d57743b30ca50b1480a72ab41a0564f20f183e92)\]\[[Code](https://github.com/TradeMaster-NTU/TradeMaster)\]
* (ICASSP 2022) Hypergraph-Based Reinforcement Learning for Stock Portfolio Selection \[[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9747138)\]\[[Code](https://github.com/Linlinmm/Stock-Portfolio-Management)\]
* (NeurIPS workshop 2022) FinRL-Meta: Market Environments and Benchmarks for Data-Driven Financial Reinforcement Learning \[[Paper](https://openreview.net/pdf?id=LkAFwrqdRY6)\]\[[Code](https://github.com/AI4Finance-Foundation/FinRL-Meta)\]
* (2022) Safe-FinRL: A Low Bias and Variance Deep Reinforcement Learning Implementation for High-Freq Stock Trading \[[Paper](https://arxiv.org/abs/2206.05910)\]\[[Code](https://github.com/Tsedao/Safe-FinRL)\]
* (NAACL 2021) Quantitative Day Trading from Natural Language using Reinforcement Learning \[[Paper](https://aclanthology.org/2021.naacl-main.316)\]\[[Code](https://github.com/midas-research/profit-naacl)\]
* (AAAI 2021) Universal Trading for Order Execution with Oracle Policy Distillation \[[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/16083)\]\[[Code](https://github.com/TradeMaster-NTU/TradeMaster)\]
* (AAAI 2021) DeepTrader: A Deep Reinforcement Learning Approach for Risk-Return Balanced Portfolio Management with Market Conditions Embedding \[[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/16144)\]\[[Code](https://github.com/TradeMaster-NTU/TradeMaster)\]
* (ICAIF 2021) FinRL: deep reinforcement learning framework to automate trading in quantitative finance \[[Paper](https://dl.acm.org/doi/10.1145/3490354.3494366)\]\[[Code](https://github.com/AI4Finance-Foundation/FinRL)\]
* (AAAI 2020) Reinforcement-Learning based Portfolio Management with Augmented Asset Movement Prediction States \[[Paper](https://arxiv.org/abs/2002.05780)\]\[[Code](https://github.com/TradeMaster-NTU/TradeMaster)\]
* (IJCAI 2020) An End-to-End Optimal Trade Execution Framework based on Proximal Policy Optimization \[[Paper](https://www.ijcai.org/Proceedings/2020/627?msclkid=a2b6ad5db7ca11ecb537627a9ca1d4f6)\]\[[Code](https://github.com/TradeMaster-NTU/TradeMaster)\]
* (ESWA 2020) Time-driven feature-aware jointly deep reinforcement learning for financial signal representation and algorithmic trading \[[Paper](https://www.sciencedirect.com/science/article/pii/S0957417419305822?via%3Dihub)\]\[[Code](https://github.com/Lingfeng158/TFJ-DRL-Replication)\]
* (2017) A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem \[[Paper](https://arxiv.org/abs/1706.10059)\]\[[Code](https://github.com/TradeMaster-NTU/TradeMaster)\]





## Meta-learning

* (KDD 2023) DoubleAdapt: A Meta-learning Approach to Incremental Learning for Stock Trend Forecasting \[[Paper](https://dl.acm.org/doi/pdf/10.1145/3580305.3599315)\]\[[Code](https://github.com/SJTU-Quant/qlib)\]
* (2023) Generative Meta-Learning Robust Quality-Diversity Portfolio \[[Paper](https://arxiv.org/abs/2307.07811)\]\[[Code](https://github.com/kayuksel/generative-opt)\]

## Multi-task Learning

* (2022) Quantitative Stock Investment by Routing Uncertainty-Aware Trading Experts: A Multi-Task Learning Approach \[[Paper](https://arxiv.org/abs/2207.07578)\]\[[Code](https://github.com/johnson7788/AlphaMix)\]
* (ACM MM 2020) Multimodal Multi-Task Financial Risk Forecasting \[[Paper](https://dl.acm.org/doi/10.1145/3394171.3413752)\]\[[Code](https://github.com/midas-research/multimodal-financial-forecasting)\]
## Contrastive Learning

* (CIKM 2021) Stock Trend Prediction with Multi-granularity Data: A Contrastive Learning Approach with Adaptive Fusion \[[Paper](http://staff.ustc.edu.cn/~cheneh/paper_pdf/2021/Min-Hou-CIKM.pdf)\]\[[Code](https://github.com/CMLF-git-dev/CMLF)\]

## Curriculum Learning

* (UAI 2021) Modeling Financial Uncertainty with Multivariate Temporal Entropy-based Curriculums \[[Paper](https://www.auai.org/uai2021/pdf/uai2021.638.preliminary.pdf)\]\[[Code](https://github.com/midas-research/finclass-uai)\]

## Ensemble Learning

* (ICDM 2020) DoubleEnsemble: A New Ensemble Method Based on Sample Reweighting and Feature Selection for Financial Data Analysis \[[Paper](https://www.semanticscholar.org/paper/DoubleEnsemble%3A-A-New-Ensemble-Method-Based-on-and-Zhang-Li/96d8383288eba50d69f516522154cf52625c7a4f)\]\[[Code](https://github.com/Sakura-Fire-Capital/DoubleEnsembleML)\]
