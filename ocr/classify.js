class Classifier {

    constructor() {

        this.loadmodel();

    }

    //loading model
    async loadmodel() {

        this.model = await tf.loadModel('model/model.json');
        
    }


    //get the result
    async evaluate(data) {
        console.log(data);
        const val = this.model.predict(data);
        console.log(val);
        val.data().then(function (value) {
            console.log(value);
        });

    }

}
