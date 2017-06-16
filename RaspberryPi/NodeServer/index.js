var express    = require('express')
var app        = express()
var bodyParser = require('body-parser')

app.use(bodyParser.json())
var port = process.env.PORT || 8080


var router = express.Router()

router.get('/', function(req, res) {
    res.json({ message: 'This is awesome huh!!' })
})

router.post('/control', function(req, res) {
	console.log('Received: ', req.body)
})

app.use('/', router)

app.listen(port)
console.log('Magic happens on port' + port)
