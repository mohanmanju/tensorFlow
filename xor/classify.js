class Classifier {

    constructor() {

        this.loadmodel();

    }

    //loading model
    async loadmodel() {

        this.model = await tf.loadModel('../../model.json');
        
    }


    //get the result
    async evaluate(data) {

        const val = this.model.predict(data);

        val.data().then(function (value) {
            result.elt.innerHTML = Math.round(value[0]);
        });

    }

}