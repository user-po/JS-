import * as tfvis from '@tensorflow/tfjs-vis';
import * as tf from '@tensorflow/tfjs';

window.onload = async ()=>{

    const heights = [150,160,170];
    const weights  = [40,50,60];

    tfvis.render.scatterplot(
        {name:'身高体重训练数据'},
        {values:heights.map((x,i)=>({x,y:weights[i]}))},
        {
            xAxisDomain:[140,180],
            yAxisDomain:[30,70]
        }
    );
   //数据归一化
    const inputs =  tf.tensor(heights).sub(150).div(20);
    const labels = tf.tensor(weights).sub(40).div(20);

     //初始化模型
    // 连续的模型 这一层的输入一定是上一层的输出
    const model = tf.sequential()
    //添加一层 全链接层 x乘一个再加一个
    //1个神经元 dense线性回归常用的神经网络This layer implements the operation: output = activation(dot(input, kernel) + bias)
    model.add(tf.layers.dense({units:1,inputShape:[1]}))
    //设置损失函数均方误差 设置随机梯度下降的减少损失方法
    model.compile({loss:tf.losses.meanSquaredError,optimizer:tf.train.sgd(0.1)})
   
    await model.fit(inputs,labels,{
        batchSize: 3,
        epochs:200,
        callbacks:tfvis.show.fitCallbacks(
            {name: '训练过程'},
            //想看什么的图像 损失函数的
            ['loss']
            )
    })

    const output = model.predict(tf.tensor([180]).sub(150).div(20))
    //反归一化
    alert('如果身高为180cm，那么预测体重为'+output.mul(20).add(40).dataSync()[0]+'kg')

}