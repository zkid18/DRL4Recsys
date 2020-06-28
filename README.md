# Deep Reinforcement Learning for Recommender Systems 
Courses on Deep Reinforcement Learning (DRL) and DRL papers for recommender system

## Courses
#### UCL Course on RL 
[http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html)
#### CS 294-112 at UC Berkeley
[http://rail.eecs.berkeley.edu/deeprlcourse/](http://rail.eecs.berkeley.edu/deeprlcourse/)
#### Stanford CS234: Reinforcement Learning
[http://web.stanford.edu/class/cs234/index.html](http://web.stanford.edu/class/cs234/index.html)

## Book
1. **Reinforcement Learning: An Introduction (Second Edition)**. Richard S. Sutton and Andrew G. Barto. [book](http://incompleteideas.net/book/bookdraft2017nov5.pdf)

## Papers
### Survey Papers
1. **A Brief Survey of Deep Reinforcement Learning**. Kai Arulkumaran, Marc Peter Deisenroth, Miles Brundage, Anil Anthony Bharath. 2017. [paper](https://arxiv.org/pdf/1708.05866.pdf)
1. **Deep Reinforcement Learing: An Overview**. Yuxi Li. 2017. [paper](https://arxiv.org/pdf/1701.07274.pdf)

### Conference Papers
1. **An MDP-Based Recommender System**. Guy Shani, David Heckerman, Ronen I. Brafman. JMLR 2005. [paper](http://www.jmlr.org/papers/volume6/shani05a/shani05a.pdf)
1. **Usage-Based Web Recommendations: A Reinforcement Learning Approach**. Nima Taghipour, Ahmad Kardan, Saeed Shiry Ghidary. Recsys 2007. [paper](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.157.9640&rep=rep1&type=pdf)
1. **DJ-MC: A Reinforcement-Learning Agent for Music Playlist Recommendation**. Elad Liebman, Maytal Saar-Tsechansky, Peter Stone. AAMAS 2015. [paper](https://arxiv.org/pdf/1401.1880.pdf)
1. **Learning to Collaborate: Multi-Scenario Ranking via Multi-Agent Reinforcement Learning**. Jun Feng, Heng Li, Minlie Huang, Shichen Liu, Wenwu Ou, Zhirong Wang, Xiaoyan Zhu. WWW 2018. [paper](https://arxiv.org/pdf/1809.06260.pdf)
1. **Reinforcement Mechanism Design for e-commerce**. Qingpeng Cai, Aris Filos-Ratsikas, Pingzhong Tang, Yiwei Zhang. WWW 2018. [paper](https://arxiv.org/pdf/1708.07607.pdf)
1. **DRN: A Deep Reinforcement Learning Framework for News Recommendation**. Guanjie Zheng, Fuzheng Zhang, Zihan Zheng, Yang Xiang, Nicholas Jing Yuan, Xing Xie, Zhenhui Li. WWW 2018. [paper](http://www.personal.psu.edu/~gjz5038/paper/www2018_reinforceRec/www2018_reinforceRec.pdf)
1. **Deep Reinforcement Learning for Page-wise Recommendations**. Xiangyu Zhao, Long Xia, Liang Zhang, Zhuoye Ding, Dawei Yin, Jiliang Tang.  RecSys 2018. [paper](https://arxiv.org/pdf/1805.02343.pdf)
1. **Recommendations with Negative Feedback via Pairwise Deep Reinforcement Learning**. Xiangyu Zhao, Liang Zhang, Zhuoye Ding, Long Xia, Jiliang Tang, Dawei Yin. KDD 2018. [paper](https://arxiv.org/pdf/1802.06501.pdf)
1. **Stabilizing Reinforcement Learning in Dynamic Environment with Application to Online Recommendation**. Shi-Yong Chen, Yang Yu, Qing Da, Jun Tan, Hai-Kuan Huang, Hai-Hong Tang. KDD 2018. [paper](http://lamda.nju.edu.cn/yuy/GetFile.aspx?File=papers/kdd18-RobustDQN.pdf)
1. **Reinforcement Learning to Rank in E-Commerce Search Engine: Formalization, Analysis, and Application**. Yujing Hu, Qing Da, Anxiang Zeng, Yang Yu, Yinghui Xu. KDD 2018. [paper](https://arxiv.org/pdf/1803.00710.pdf)
1. **A Reinforcement Learning Framework for Explainable Recommendation**. Xiting Wang, Yiru Chen, Jie Yang, Le Wu, Zhengtao Wu, Xing Xie. ICDM 2018. [paper](https://www.microsoft.com/en-us/research/uploads/prod/2018/08/main.pdf) [video] (https://youtu.be/Ys3YY7sSmIA)

#### **Top-K Off-Policy Correction for a REINFORCE Recommender System**. Minmin Chen, Alex Beutel, Paul Covington, Sagar Jain, Francois Belletti, Ed H. Chi. WSDM 2019. [paper](https://arxiv.org/pdf/1812.02353.pdf)

- Leverage on a policy-based algorithm, REINFORCE.
- Live experiment on Youtube data.
- State - user interest, context
- Action - items available for recommendations. Constrain by sampling each item according to the softmax policy.
- Reward - clicks, watch time. As the recommender system lists the page of k-items to a user in time, the expected reward of a set equal to the sum of an expected reward of each item in the set.
- Discount rate - the trade-off between an immediate reward and long-term reward
- Rely on the logged feedback of auctions chosen by a historical policy (or a mixture of policies) as it's not possible to perform online updates of the policy.
- Model state transition with a recurrent neural network (tested LSTM, GRU, Chaos Free RNN) <img src="https://render.githubusercontent.com/render/math?math=s_{t+1} = f(s_t, u_{a_t})"> 
- Feedback is biased by previous recommender. Policy not update frequently that also cause bias. Feedback could come from other agents in production.
- Agent is refreshed every 5 days
- Used target (on-policy) and behavioral policy (off-policy) to downweitgth non-target policies.
- The epsilon-greedy policy is not acceptable for Youtube as it causes inappropriate recommendations and bad user experience. Employ Boltzman exploration.
- Target policy is trained with weighted softmax to max long term reward.
- Behavior policy trained using state/action pairs from logged behavior/feedback.
- On an online experiment, the reward is aggregated on a time horizon of 4-10 hours.
- Parameters and architecture are primarily shared.
- RNN is used to represent the user state
- Private dataset
- Code reproduction by Bayes group [code] (https://github.com/awarebayes/RecNN)

<img src="https://user-images.githubusercontent.com/8243154/84734543-d4a2e300-afa9-11ea-9263-16e2cef1d9f8.png">

1. **Generative Adversarial User Model for Reinforcement Learning Based Recommendation System**. Xinshi Chen, Shuang Li, Hui Li, Shaohua Jiang, Yuan Qi, Le Song. ICML 2019. [paper](http://proceedings.mlr.press/v97/chen19f/chen19f.pdf)
1. **Aggregating E-commerce Search Results from Heterogeneous Sources via Hierarchical Reinforcement Learning**. Ryuichi Takanobu, Tao Zhuang, Minlie Huang, Jun Feng, Haihong Tang, Bo Zheng. WWW 2019. [paper](https://arxiv.org/pdf/1902.08882.pdf)
1. **Policy Gradients for Contextual Recommendations**. Feiyang Pan, Qingpeng Cai, Pingzhong Tang, Fuzhen Zhuang, Qing He. WWW 2019. [paper](https://arxiv.org/pdf/1802.04162.pdf)
1. **Reinforcement Knowledge Graph Reasoning for Explainable Recommendation**. Yikun Xian, Zuohui Fu, S. Muthukrishnan, Gerard de Melo, Yongfeng Zhang. SIGIR 2019. [paper](http://yongfeng.me/attach/xian-sigir2019.pdf)

#### **Reinforcement Learning to Optimize Long-term User Engagement in Recommender Systems**. Lixin Zou, Long Xia, Zhuoye Ding, Jiaxing Song, Weidong Liu, Dawei Yin. KDD 2019. [paper](https://arxiv.org/pdf/1902.05570.pdf)
- Introduced an RL framework - FeedRec to optimize long-term user engagement. Based on Q-network which designed in hierarchical LSTM and S-network, that simulates the environment 
- S-network assists Q-Network and voids the instability of convergence in policy learning. Specifically, in each round of recommendations, aligning with the user feedback, S-network generates the user's response, the dwell time, the revisited time, and flag that indicates that the user will leave or not the platform.
- The model versatile both instant (cliks, likes, ctr) and delayed (dweel time, revisit and etc)
- Trained on internal e-commerce dataset with pre-trained embeddings, which is learned through modeling user's cliking streams with skip-gram.
- State - the user's browsing history
- Action - the finite space of items
- Transition - probability of seeing state <img src="https://render.githubusercontent.com/render/math?math=s_{t+1}"> after taking action <img src="https://render.githubusercontent.com/render/math?math=i_t"> at  <img src="https://render.githubusercontent.com/render/math?math=s_t"> 
- Reward as a weighted sum of different metrics. Give some instantiations of reward function, both instant and delayed metrics.
- The primary user behaviors, such as clicks, skip, the purchase is tracked separately with different LSTM pipelines as where different user's behavior is captured by the corresponding LSTM-layer to avoid intensive behavior dominance and capture specific characteristics.
<img src="https://user-images.githubusercontent.com/8243154/84734503-b3da8d80-afa9-11ea-80f9-bd57a9afa668.png">

1. **Environment reconstruction with hidden confounders for reinforcement learning based recommendation**. Wenjie Shang, Yang Yu, Qingyang Li, Zhiwei Qin, Yiping Meng, Jieping Ye. KDD 2019. [paper](http://lamda.nju.edu.cn/yuy/GetFile.aspx?File=papers/kdd19-confounder.pdf)
1. **Exact-K Recommendation via Maximal Clique Optimization**. Yu Gong, Yu Zhu, Lu Duan, Qingwen Liu, Ziyu Guan, Fei Sun, Wenwu Ou, Kenny Q. Zhu. KDD 2019. [paper](https://arxiv.org/pdf/1905.07089.pdf)
1. **Hierarchical Reinforcement Learning for Course Recommendation in MOOCs**. Jing Zhang, Bowen Hao, Bo Chen, Cuiping Li, Hong Chen, Jimeng Sun. AAAI 2019. [paper](https://xiaojingzi.github.io/publications/AAAI19-zhang-et-al-HRL.pdf)
1. **Large-scale Interactive Recommendation with Tree-structured Policy Gradient**. Haokun Chen, Xinyi Dai, Han Cai, Weinan Zhang, Xuejian Wang, Ruiming Tang, Yuzhou Zhang, Yong Yu. AAAI 2019. [paper](https://arxiv.org/pdf/1811.05869.pdf)
#### **Virtual-Taobao: Virtualizing real-world online retail environment for reinforcement learning**. Jing-Cheng Shi, Yang Yu, Qing Da, Shi-Yong Chen, An-Xiang Zeng. AAAI 2019. [paper](http://www.lamda.nju.edu.cn/yuy/GetFile.aspx?File=papers/aaai2019-virtualtaobao.pdf)
- Instead of training reinforcement learning in Taobao directly, we present our approach: first build Virtual Taobao, a simulator learned from historical customer behavior data through the proposed GAN-SD (GAN for Simulating Distributions) and MAIL (multi-agent adversaria limitation learning), and then we train policies in Virtual Taobao with no physical costs in which ANC (Action Norm Constraint) strategy is proposed to reduce over-fitting.
- Comparing with the traditional supervised learning approach, the strategy trained in Virtual Taobao achieves more than 2% improvement of revenue in the real environment.
- Agent - the search engine
- Environment - the customers
- Commodity search and shopping process can see as a sequential decision process. Customers decision process for engine and customers. The engine and customers are the environments of each other

1. **A Model-Based Reinforcement Learning with Adversarial Training for Online Recommendation**. Xueying Bai, Jian Guan, Hongning Wang. NeurIPS 2019. [paper](http://papers.nips.cc/paper/9257-a-model-based-reinforcement-learning-with-adversarial-training-for-online-recommendation.pdf)
1. **Text-Based Interactive Recommendation via Constraint-Augmented Reinforcement Learning**. Ruiyi Zhang, Tong Yu, Yilin Shen, Hongxia Jin, Changyou Chen, Lawrence Carin. NeurIPS 2019. [paper](http://people.ee.duke.edu/~lcarin/Ruiyi_NeurIPS2019.pdf)
1. **DRCGR: Deep reinforcement learning framework incorporating CNN and GAN-based for interactive recommendation**. Rong Gao, Haifeng Xia, Jing Li, Donghua Liu, Shuai Chen, and Gang Chun. ICDM 2019. [paper](https://ieeexplore.ieee.org/document/8970700)
1. **Pseudo Dyna-Q: A Reinforcement Learning Framework for Interactive Recommendation**. Lixin Zou, Long Xia, Pan Du, Zhuo Zhang, Ting Bai, Weidong Liu, Jian-Yun Nie, Dawei Yin. WSDM 2020. [paper](https://tbbaby.github.io/pub/wsdm20.pdf)
1. **End-to-End Deep Reinforcement Learning based Recommendation with Supervised Embedding**. Feng Liu, Huifeng Guo, Xutao Li, Ruiming Tang, Yunming Ye, Xiuqiang He. WSDM 2020. [paper](https://dl.acm.org/doi/abs/10.1145/3336191.3371858)
1. **Reinforced Negative Sampling over Knowledge Graph for Recommendation**. Xiang Wang, Yaokun Xu, Xiangnan He, Yixin Cao, Meng Wang, Tat-Seng Chua. WWW 2020. [paper](https://arxiv.org/pdf/2003.05753.pdf)

#### **Deep Reinforcement Learning for List-wise Recommendations** Xiangyu Zhao, Liang Zhang, Long Xia, Zhuoye Ding, Dawei Yin, Jiliang Tang, 2017 [paper](arxiv.org/abs/1801.00209)

- Utilized the Actor-critic framework to work with the increasing number of items for recommendations.
- Proposed an online environment simulator to pre-train parameters offline and evaluate the model before applying.
- Recommender agent interacts with the environment (users) choosing items over a sequence of steps.
- State - browsing history of the user, i.e., the previous N items that a user browsed before. Users browsing history stored in a memory M. However a better way is to consider only N previous clicked/ordered items. The positive items represent key information about user's preference.
- Action - a list of items recomended to a user at time t based on the current state.
- Reward - clicks, order, etc
- Transition probability - Probability of state transition from <img src="https://render.githubusercontent.com/render/math?math=s_t"> to <img src="https://render.githubusercontent.com/render/math?math=s_{t+1}">

- Each time observe a K items in temporal order.
- Utilized DDPG algorithm with experienced decay to train the parameters of the framework.
- MAP and NDCG to evaluate performance.
- No code
- Private dataset

#### **Deep Reinforcement Learning based Recommendation with Explicit User-Item Interactions Modeling** Feng Liu, Ruiming Tangy, Xutao Li, Weinan ZhangzYunming Ye, Haokun Chenz, Huifeng Guoyand Yuzhou Zhangy, 2019 [paper](arxiv.org/abs/1810.12027)

- The DRR framework treats recommendations as a sequential decision making procedure and adopts "Actor-critic" reinforcement-learning scheme to model interactions between the user and recommender items. The framework treats recommendations as a sequential decision-making process, which consider both immediate and long-term reward.

- State - User's positive interaction history with recommender as well as her demographic situation.
- Actions - continuous parameter vector a. Each item has a ranking score, which is defined as the inner product of the action and the item embedding.
- Transitions - once the user's feedback is collected the transition is determined
- Reward - clicks, rates, etc
- Discount rate - the trade-off between immediate reward and long-term reward

<img src ="https://user-images.githubusercontent.com/8243154/84734575-edab9400-afa9-11ea-8199-4f4a1b34b7d9.png">

- Developed three state-representation models:
  - DRR-p Utilize the product operator to capture the pairwise local dependencies between items. Compute pairwise interactions between items by using the element-wise product operations.
  - DRR-u In addition to local dependencies between items, the pairwise interactions of user-item are also taken into account.
  - DRR-ave. Average pooling over DRR-p. The resulting vector is leveraged to model the interactions with the input user. Finally, concat the interaction vector and embedding of the user. On offline evaluation, DRR-ave outperformed DRR-p and DRR-u.

- Epsilon-greedy exploration in the Actor-network.
- Evaluated on offline datasets (MovieLens, Yahoo! Music, Jester)
- Online evaluation with environment simulator. Pretrain PMF (probabilistic matrix factorization) model as an environment simulator.

#### **Reinforcement Learning for Slate-based RecommenderSystems: A Tractable Decomposition and Practical Methodology**. Eugene Ie†, Vihan Jain, Jing Wang, Sanmit Narvekar, Ritesh Agarwal1, Rui Wu1, Heng-Tze Cheng1, Morgane Lustman, Vince Gatto3, Paul Covington, Jim McFadden, Tushar Chandra, and Craig Boutilier† [paper] (https://arxiv.org/pdf/1905.12767.pdf) [video] (https://www.youtube.com/watch?v=X3uozqaNCYE)

- Developed a SLATEQ, a decompositio nof value-based temporal-difference and Q-learning that renders RL tractable with slates
- LTV of a slate can be decomposed into a tractable value function of its component item-wise LTV
- Introduced a recommender simulation environment, RecSim that allows the straightforward configuration of an itemcollection (or vocabulary), a user (latent) state model and a user choice model.
- Session optimization
- State - static users features as demographics, declared intresets, user context, summarizaiton of the previous history
- Acions - the set of all possible slates.
- Transition probability
- Reward - measurement of user engagement. 
- In the MDP model each user should be viewed as a separate environment or separate MDP. Hence it critical to allow generalization across users, since few if any users generates enough experience to allow reasonable recommendations.
- Combinatorial optimizaton problem - find the slate with the maximum Q-value. SlateQ allows to decompose into cobination of the item-wise Q-values of the consistent items. Tried top-k, greedy, LP-based methods.
- Items and user intersts as a topic modeling problem
- As a choice model, use an exponential cascade model that accounts for document position in the slate. This choice model assumes "attention" is given to one document at a time, with exponentially decreasing attention given to documents as a user moves down the slate.
- A user choice model impacts which document(if any) from the slate is consumed by the user. Made an assumption that a user can observe any recommender document's topic before selection but can't observe its quality before consumption
- Users interested in document d defines the relative document appeal to the user and serves the basis of the choice function.
- Models are trained periodically and pushed to the server. The ranker uses the latest model to recommend items and logs user feedback, which is used to train new models. Using LTV labels, iterative model training, and pushing can be viewed as a form of generalized policy iteration.
- A candidate generator retrieves a small subset (hundreds) of items from a large corpus that best matches a user context. Therankerscores/ranks are candidates using a DNN with both user context and item features as input. It optimizes a combination of several objectives (e.g., clicks, expected engagement, several other factors).

#### **RECSIM : A Configurable Simulation Platform for Recommender Systems, Ie E, Hsu C, Mladenov M, Jain V, Narvekar S, Wang J, Wu R, Boutilier C
2019** [paper] (https://arxiv.org/pdf/1909.04847.pdf)
- RECSIM is a configurable platform that allows the natural, albeit abstract, specification of an environment in which a recommender interacts with a corpus of documents (or recommendable items) and a set of users, to support the development of recommendation algorithms.
- The user model samples users from a prior distribution over (configurable) user features: these may include latent features
such as personality, satisfaction, interests; observable features such as demographics; and behavioral features such as session length, visit frequency, or time budget
- The document model samples items from a prior over document features, which again may incorporate latent features such as document quality, and observable features such as topic, document length and global statistics (e.g., ratings, popularity)
- User responce is determined by user choice model. Once the document is consumed the user state undergoes a transition through a cinfigurable (user) transition model
- The user model assumes that user have a various degrees of interests in topic (with some prior distribution) ranking from -1 to 1.
<img src = "https://user-images.githubusercontent.com/8243154/85855204-8c13d280-b7be-11ea-8344-e02d2be9d75f.png">
- RECSIM offers stackable hierarchical agent layers intended to solve a more abstract recommendation problem. A hierarchical agent layer does not materialize a slate of documents (i.e., RS action), but relies on one or more base agents to do so.
- RECSIM can be viewed as a dynamic Bayesian network that defines a probability distribution over trajectories of slates, choices, and observations.
- Several baseline agents implemented:
-- TabularQAgent - Q-learning algorithm. Enumarates all possible slates to maximize over Q values
-- Full-slateQAgent - deep Q-Network agent by threating each slate as a single action
-- Random agent - random slates with no duplicates
- [RECSIM code] (https://github.com/google-research/recsim)
<img src = "https://user-images.githubusercontent.com/8243154/85855273-ac439180-b7be-11ea-8942-b82dfaed71c9.png">


### Preprint Papers
1. **Reinforcement Learning based Recommender System using Biclustering Technique**. Sungwoon Choi, Heonseok Ha, Uiwon Hwang, Chanju Kim, Jung-Woo Ha, Sungroh Yoon. arxiv 2018. [paper](https://arxiv.org/pdf/1801.05532.pdf) 
1. **Deep Reinforcement Learning based Recommendation with Explicit User-Item Interactions Modeling**. Feng Liu, Ruiming Tang, Xutao Li, Weinan Zhang, Yunming Ye, Haokun Chen, Huifeng Guo, Yuzhou Zhang. arxiv 2018. [paper](https://arxiv.org/pdf/1810.12027.pdf)
1. **Model-Based Reinforcement Learning for Whole-Chain Recommendations**. Xiangyu Zhao, Long Xia, Yihong Zhao, Dawei Yin, Jiliang Tang. arxiv 2019. [paper](https://arxiv.org/pdf/1902.03987.pdf)
