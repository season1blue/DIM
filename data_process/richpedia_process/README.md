richpedia数据集不能直接使用划分好的，需要从原始的richpedia-MEL中划分，所以数据处理部分需要加一个split

执行build_text 构建所需的文本信息
然后使用neg_clip_extract，用clip提取fact.json中的brief信息作为ground_truth