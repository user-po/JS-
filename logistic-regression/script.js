import * as tfvis from '@tensorflow/tfjs-vis';
import * as tf from '@tensorflow/tfjs'
import {getData} from './data.js'
window.onload = async ()=>{
 const data = getData(400);
 
 tfvis.render.scatterplot(
     {name:'逻辑回归训练数据'},
     {
         values: [
             data.filter(p=>p.label === 1),
             data.filter(p=>p.label === 0),
         ]
     }
 )

 //1.初始化神经网络模型
 const model = tf.sequential()
 //2.为神经网络添加层 神经元数量 神经元输入的形状[2,3]
//3. 设计层的神经元个数 特征数量(inputShape) 设置激活函数
 model.add(tf.layers.dense({
     units:1,
     inputShape:[2],
     activation:'sigmoid'
 }))
 //4.设置损失函数
 model.compile({loss:tf.losses.logLoss,optimizer: tf.train.adam(0.1)})

 //5.特征数量为2的数据转化为tensor
 const inputs = tf.tensor(data.map(p=>[p.x,p.y]))
 const labels = tf.tensor(data.map(p=>p.label))

 await model.fit(inputs,labels,{
     batchSize:40,
     epochs: 50,
     callbacks:tfvis.show.fitCallbacks(
         {name:'训练过程'},
         ['loss']
     )
 })
 window.predict = (form)=>{
     const pred = model.predict(tf.tensor([[form.x.value * 1,form.y.value*1]]))
     alert(`预测结果：${pred.dataSync()[0]}`)
 }
}