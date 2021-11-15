((function() {
    var callbacks = [],
        timeLimit = 50,
        open = false;
    setInterval(loop, 1);
    return {
        addListener: function(fn) {
            callbacks.push(fn);
        },
        cancleListenr: function(fn) {
            callbacks = callbacks.filter(function(v) {
                return v !== fn;
            });
        }
    }

    function loop() {
        var startTime = new Date();
        //打开控制台后就会使得时间间隔大于50ms
        debugger;
        if (new Date() - startTime > timeLimit) {
            if (!open) {
                callbacks.forEach(function(fn) {
                    fn.call(null);
                });
            }
            open = true;
            window.stop();
            alert('极锋网提醒大哥不要扒我了，劳烦您尊重一下劳动成果！');
            document.body.innerHTML = "";
        } else {
            open = false;
        }
    }
})()).addListener(function() {
    window.location.reload();
});