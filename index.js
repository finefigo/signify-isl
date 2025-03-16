const express = require('express');
const bodyParser = require('body-parser');
const { PythonShell } = require('python-shell');

const app = express();
app.use(bodyParser.json());

// Route to handle sign language recognition
app.post('/recognize-sign', (req, res) => {
    const imageData = req.body.imageData; // Image data from the frontend

    // Options to pass to the Python script
    let options = {
        args: [imageData]
    };

    PythonShell.run('recognize_sign.py', options, (err, results) => {
        if (err) res.status(500).send(err);
        res.send({ gestureText: results[0] });
    });
});

app.listen(3000, () => {
    console.log('Server is running on port 3000');
});
