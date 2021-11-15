import * as tfvis from "@tensorflow/tfjs-vis";
import * as tf from "@tensorflow/tfjs";
import { MnistData } from './data'

window.onload = async ()=>{
    //读取图片和label
    const data = new MnistData();
    //加载图片和二进制label图片 图片转换为tensor 
    await data.load()
    const examples = data.nextTestBatch(20);
    const surface = tfvis.visor().surface({name:'输入数据'})
    for(let i=0;i<20;i++){
         const imageTensor = tf.tidy(()=>{
           return  examples.xs.slice([i,0],[1,784]).reshape([28,28,1]);

         })

         const canvas = document.createElement('canvas');
         canvas.width = 28;
         canvas.height = 28;
         canvas.style = 'margin:4px'
         await tf.browser.toPixels(imageTensor,canvas)
         surface.drawArea.appendChild(canvas)
    }
    const model = tf.sequential();
    //两轮提取
    model.add(tf.layers.conv2d({
        inputShape:[28,28,1],
        kernelSize:3,
        filters:8,
        strides:1,
        activation:'relu',
        kernelInitializer:'varianceScaling'
    }));

    //最大池化层
    model.add(tf.layers.maxPool2d({
        poolSize:[2,2],
        strides:[2,2]
    }))

    model.add(tf.layers.conv2d({
         kernelSize:5,
         filters:16,
         strides:1,
         activation:'relu',
         kernelInitializer:'varianceScaling'
    }))
    model.add(tf.layers.maxPool2d({
        poolSize:[2,2],
        strides:[2,2]
    }));
    //提取出的高维特征转化为一维
    model.add(tf.layers.flatten()); 
    model.add(tf.layers.dense({
        // 数字有10个分类
        units:10,
        activation:'softmax',
        kernelInitializer:'varianceScaling'

    }))

    model.compile({
        loss:'categoricalCrossentropy',
        optimizer:tf.train.adam(),
        metrics:'accuracy'
    });
   const [trainXs,trainYs] = tf.tidy(()=>{
       const d = data.nextTrainBatch(5000); 
       return [
        //    和输入形状一样
           d.xs.reshape([5000,28,28,1]),
           d.labels,
       ]
   })
   const [testXs,testYs] = tf.tidy(()=>{
    const d = data.nextTestBatch(200); 
    return [
     //    和输入形状一样
        d.xs.reshape([200,28,28,1]),
        d.labels,
    ]
})

await model.fit(trainXs,trainYs,{
      validationData:[testXs,testYs],
      epochs:50,
      callbacks: tfvis.show.fitCallbacks(
          {name:'训练效果'},
          ['loss','val_loss','acc','val_acc'],
          {callbacks:['onEpochEnd']}
      )
})
    const canvas = document.querySelector('canvas')

    canvas.addEventListener('mousemove',(e)=>{
        if(e.buttons ===1){
            const ctx = canvas.getContext('2d');
            ctx.fillStyle = 'rgb(255,255,255)';
            ctx.fillRect(e.offsetX,e.offsetY,10,10)
        }
    })
    window.clear = ()=>{
         const ctx = canvas.getContext('2d');
         ctx.fillStyle = 'rgb(0,0,0)';
         ctx.fillRect(0,0,300,300)
    }
    clear();

    window.predit = ()=>{
        const input = tf.tidy(()=>{
            //图片变为28 28
            return tf.image.resizeBilinear(
                //canvas转tensor
                tf.browser.fromPixels(canvas),
                [28,28],
                true
            ) //变成黑白图片
            .slice([0,0,0],[28,28,1])
            //归一化
            .toFloat()
            .div(255)
            //shape变成28 28 1
            .reshape([1,28,28,1])
        })

        const pred = model.predict(input).argMax(1);
    
        alert(`手写的数字为${pred.dataSync()[0]}`)
    }

}