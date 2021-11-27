# LAB 1

### 实验介绍和实验目的

ModelArts自动学习是帮助人们实现AI应用的低门槛、高灵活、零代码的定制化模型开发工具。自动学习功能根据标注数据自动设计模型、自动调参、自动训练、自动压缩和部署模型。当前自动学习支持快速创建图像分类、物体检测、预测分析和声音分类模型的定制化开发。可广泛应用在工业、零售安防等领域。

本实验通过ModelArts自动学习实现动物叫声分类，主要面向业务开发者，无需专业的开发基础和编码能力，只需上传数据，通过自动学习界面引导和简单操作即可完成模型训练和部署。

本次实验目的为熟练使用ModelArts自动学习实现模型训练与部署。

### 实验步骤

1. 在ModelArts的自动学习界面创建“声音分类”项目。	

   ![image-20211127162048331](https://gitee.com/shotray/img-host/raw/master/20211127162056.png)

2. 将数据集中的音频数据进行上传，并进行分类标识。

   训练集中共有100条数据，共四个标签，通过手动选择进行标签的分配，可知每个标签中有25个语音项目。

   ![image-20211127162303900](https://gitee.com/shotray/img-host/raw/master/20211127162305.png)

3. 在标注完成后对模型进行训练，其训练结果如下图所示。

   ![image-20211127163744289](https://gitee.com/shotray/img-host/raw/master/20211127163745.png)

   可以看出本次训练一共训练了一分十三秒，其各参数如下表所示

   | 评估参数 | 评估结果 |
   | -------- | -------- |
   | 召回率   | 0.950    |
   | 精确率   | 0.958    |
   | 准确率   | 0.950    |
   | F1值     | 0.949    |

   对于不同标签的训练，其参数和结果如下表所示。

   | 标签名 | F1值  | 精确率 | 召回率 |
   | ------ | ----- | ------ | ------ |
   | tiger  | 0.909 | 0.833  | 1.000  |
   | bird   | 1.000 | 1.000  | 1.000  |
   | dog    | 1.000 | 1.000  | 1.000  |
   | cat    | 0.889 | 1.000  | 0.800  |

4. 对训练好的模型进行部署，部署后可以进行模型的测试。

   ![image-20211127164551547](https://gitee.com/shotray/img-host/raw/master/20211127164552.png)

5. 部署好的模型如下图所示。

   ![image-20211127165814292](https://gitee.com/shotray/img-host/raw/master/20211127165815.png)

6. 我们分别对四个标签的音频各挑选出一个进行预测。

   - bird

   ![image-20211127165944546](https://gitee.com/shotray/img-host/raw/master/20211127165945.png)

   - cat

   ![image-20211127170403135](https://gitee.com/shotray/img-host/raw/master/20211127170404.png)

   - tiger

   ![image-20211127170427761](https://gitee.com/shotray/img-host/raw/master/20211127170428.png)
   
   - dog
   
   ![image-20211127170951241](https://gitee.com/shotray/img-host/raw/master/20211127170952.png)
   
7. 对模型进行批量测试，选择批量服务。

   ![image-20211127171535481](https://gitee.com/shotray/img-host/raw/master/20211127171536.png)

8. 对模型进行部署。

   ![image-20211127171553279](https://gitee.com/shotray/img-host/raw/master/20211127171554.png)

9. 部署后通过代码获取处理好的数据并测试accuracy

   ```python
   try:
       resp = obsClient.listObjects('bucket-6576','speechsepecification/output/infer-result-1ae2c7fa-a34a-48a5-af4c-bdbad2f88973') 
   
       if resp.status < 300: 
           all = 0
           correct = 0
           write_file = csv.DictWriter(open('test.csv','w',newline='',encoding = 'utf-8'),['file_tag','tag','score'])
           write_file.writeheader()
           for info in resp.body['contents']:
               if info['key'].endswith('txt'):
                   all += 1
                   type = info['key'].split('/')[-1].split('_')[0]
                   data = obsClient.getObject('bucket-6576',str(info['key']),loadStreamInMemory=True)
                   data = data['body']['buffer'].decode('utf-8')
                   data = json.loads(data)
                   if type == data['predicted_label']:
                       correct += 1
                   write_file.writerow({'file_tag':type,'tag': data['predicted_label'],'score': data["score"]})
           print("accuracy: ", correct / all)
       else: 
           print('errorCode:', resp.errorCode) 
           print('errorMessage:', resp.errorMessage)
   except:
       import traceback
       print(traceback.format_exc())
   ```
   其accuracy为1.0
   
10. 数据对应如下图所示。

    ![image-20211127204125295](https://gitee.com/shotray/img-host/raw/master/20211127204125.png)

### 实验小结

本次实验对ModelArts平台以及OBS对象存储进行了了解和熟悉。

### 附录

1. 混淆矩阵
    假如现在有一个二分类问题，那么预测结果和实际结果两两结合会出现如下四种情况。
    用T(True)代表正确、F(False)代表错误、P(Positive)代表1、N(Negative)代表0。先看预测结果(P|N)，然后再针对实际结果对比预测结果，给出判断结果(T|F)。按照上面逻辑，重新分配后为

  ![image-20211127165138301](https://gitee.com/shotray/img-host/raw/master/20211127165139.png)
  TP、FP、FN、TN可以理解为
  - TP：预测为1，实际为1，预测正确。
  - FP：预测为1，实际为0，预测错误。
  - FN：预测为0，实际为1，预测错误。
  - TN：预测为0，实际为0，预测正确。

2. 准确率

   首先给出**准确率(Accuracy)**的定义，即**预测正确的结果占总样本的百分比**，表达式为

   ![image-20211127165327249](https://gitee.com/shotray/img-host/raw/master/20211127165328.png)

3. 精确率

   **精确率(Precision)**是针对预测结果而言的，其含义是**在被所有预测为正的样本中实际为正样本的概率**，表达式为

   ![image-20211127165433982](https://gitee.com/shotray/img-host/raw/master/20211127165434.png)

4. 召回率

   **召回率(Recall)**是针对原样本而言的，其含义是**在实际为正的样本中被预测为正样本的概率**，表达式为

   ![image-20211127165509642](https://gitee.com/shotray/img-host/raw/master/20211127165510.png)

5. F1分数

   精确率和召回率又被叫做查准率和查全率，可以通过P-R图进行表示

   ![image-20211127165547166](https://gitee.com/shotray/img-host/raw/master/20211127165548.png)
   
    如何理解P-R(精确率-召回率)曲线呢？或者说这些曲线是根据什么变化呢？
   
   以逻辑回归举例，其输出值是0-1之间的数字。因此，如果我们想要判断用户的好坏，那么就必须定一个阈值。比如大于0.5指定为好用户，小于0.5指定为坏用户，然后就可以得到相应的精确率和召回率。但问题是，这个阈值是我们随便定义的，并不知道这个阈值是否符合我们的要求。因此为了寻找一个合适的阈值，我们就需要遍历0-1之间所有的阈值，而每个阈值都对应一个精确率和召回率，从而就能够得到上述曲线。
   
   根据上述的P-R曲线，怎么判断最好的阈值点呢？首先我们先明确目标，我们希望精确率和召回率都很高，但实际上是矛盾的，上述两个指标是矛盾体，无法做到双高。因此，选择合适的阈值点，就需要根据实际问题需求，比如我们想要很高的精确率，就要牺牲掉一些召回率。想要得到很高的召回率，就要牺牲掉一些精准率。但通常情况下，我们可以根据他们之间的平衡点，定义一个新的指标：F1分数(F1-Score)。F1分数同时考虑精确率和召回率，让两者同时达到最高，取得平衡。F1分数表达式为
   
   ![image-20211127165646874](https://gitee.com/shotray/img-host/raw/master/20211127165648.png)

**参考**：https://cloud.tencent.com/developer/article/1486764#:~:text=%E7%B2%BE%E7%A1%AE%E7%8E%87%20%28Precision%29,%E6%98%AF%E9%92%88%E5%AF%B9%E9%A2%84%E6%B5%8B%E7%BB%93%E6%9E%9C%E8%80%8C%E8%A8%80%E7%9A%84%EF%BC%8C%E5%85%B6%E5%90%AB%E4%B9%89%E6%98%AF%20%E5%9C%A8%E8%A2%AB%E6%89%80%E6%9C%89%E9%A2%84%E6%B5%8B%E4%B8%BA%E6%AD%A3%E7%9A%84%E6%A0%B7%E6%9C%AC%E4%B8%AD%E5%AE%9E%E9%99%85%E4%B8%BA%E6%AD%A3%E6%A0%B7%E6%9C%AC%E7%9A%84%E6%A6%82%E7%8E%87%20%EF%BC%8C%E8%A1%A8%E8%BE%BE%E5%BC%8F%E4%B8%BA