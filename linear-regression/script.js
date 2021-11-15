import * as tfvis from '@tensorflow/tfjs-vis'
import * as tf from '@tensorflow/tfjs'
window.onload = async ()=>{

    const xs = [1,2,3,4];
    const ys = [1,3,5,7]

    tfvis.render.scatterplot(
        {name:'线性回归训练集'},
        {values: xs.map((x,i)=>({x,y:ys[i]}))},
        {xAxisDomain:[0,5],yAxisDomain:[0,8]},
    )
    //初始化模型
    // 连续的模型 这一层的输入一定是上一层的输出
    const model = tf.sequential()
    //添加一层 全链接层 x乘一个再加一个
    //1个神经元 dense线性回归常用的神经网络This layer implements the operation: output = activation(dot(input, kernel) + bias)
    model.add(tf.layers.dense({units:1,inputShape:[1]}))
    //设置损失函数均方误差 设置随机梯度下降的减少损失方法
    model.compile({loss:tf.losses.meanSquaredError,optimizer:tf.train.sgd(0.1)})
   
    //1.转化为tensor
    const inputs = tf.tensor(xs);
    const labels = tf.tensor(ys);

    await model.fit(inputs,labels,{
        batchSize: 4,
        epochs:200,
        callbacks:tfvis.show.fitCallbacks(
            {name: '训练过程'},
            //想看什么的图像 损失函数的
            ['loss']
            )
    })

    const output = model.predict(tf.tensor([5]))
    //转成数字
    alert('如果x为5，那么y为'+output.dataSync()[0])



}