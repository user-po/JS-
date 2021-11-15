import * as tfvis from "@tensorflow/tfjs-vis";
import * as tf from "@tensorflow/tfjs";
import { getData } from "./data.js";
window.onload = async () => {
  const data = getData(400);

  tfvis.render.scatterplot(
    { name: "XOR训练数据" },
    {
      values: [
        data.filter((p) => p.label === 1),
        data.filter((p) => p.label === 0),
      ],
    }
  );
  //初始化神经网络模型并添加两个层 设计神经元个数 输入形状 激活函数
  const model = tf.sequential();

  //隐藏层
  model.add(tf.layers.dense({units:4,inputShape:[2],activation:'relu'}));
  //输出层 只有第一层需要inputShape
  model.add(tf.layers.dense(
      {
          units:1,
          activation:'sigmoid'
    }
  ))
  model.compile({
      loss:tf.losses.logLoss,
      optimizer: tf.train.adam(0.1)
  })

  const inputs  = tf.tensor(data.map(p=>[p.x,p.y]))
  const labels  = tf.tensor(data.map(p=>p.label))

  await model.fit(inputs,labels,{
       batchSize:40,
       epochs:20,
       callbacks:tfvis.show.fitCallbacks(
           {name:'训练过程'},
           ['loss'])
  })
  window.predict = (form)=>{
    const pred = model.predict(tf.tensor([[form.x.value * 1,form.y.value*1]]))
    alert(`预测结果：${pred.dataSync()[0]}`)
 }
};
