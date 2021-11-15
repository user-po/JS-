import {getIrisData,IRIS_CLASSES} from './data'
import * as tf from "@tensorflow/tfjs";
import * as tfvis from "@tensorflow/tfjs-vis";
window.onload = async ()=>{
    // 15%数据作为验证集 xTrain训练集输入特征 x输入 y标签
    const [xTrain,yTrain,xTest,yTest] = getIrisData(0.15);

    //最后一层神经元个数必须是输入类别的个数
    //softmax 输出一个和为1的三个概率
    //只有第一层设置inputShape
    const model = tf.sequential();
    model.add(tf.layers.dense(
         {
             units:10,
             inputShape:[xTrain.shape[1]],
             activation:'sigmoid'
         }
    ))
    //多分类神经网络的核心代码
    model.add(tf.layers.dense({
         units: 3,
         activation:'softmax'

    }));

    model.compile({
        loss:'categoricalCrossentropy',
        optimizer:tf.train.adam(0.1),
        metrics:['accuracy']
    });

    await model.fit(xTrain,yTrain,{
         epochs: 100,
         validationData: [xTest,yTest],
         callbacks:tfvis.show.fitCallbacks(
             {name:'训练效果'},
             ['loss','val_loss','acc','val_acc'],
             {callbacks:['onEpochEnd']}
         )
    })

    window.predict = (form) => {
        const input = tf.tensor([[
            form.a.value * 1,
            form.b.value * 1,
            form.c.value * 1,
            form.d.value * 1,
        ]]);
        const pred = model.predict(input);
        alert(`预测结果：${IRIS_CLASSES[pred.argMax(1).dataSync(0)]}`);
    };
   
}