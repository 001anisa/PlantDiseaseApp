import { useState, useEffect, useRef } from 'react';
import * as tf from '@tensorflow/tfjs'
import { argMax } from '@tensorflow/tfjs';

function App() {
    const [isModelLoading, setIsModelLoading] = useState(false) //for model loading 
    const [model, setModel] = useState(null)   //to load the model.json file in the state variable
    const [imageURL, setImageURL] = useState(null);  //used to upload image via url
    const [results, setResults] = useState([])  //to output the results of the image content
    const [history, setHistory] = useState([])  //this to maintain the recent pictures used
    const disease = ['Early Blight','Late Blight','Healthy']   //the class of the disease
    

    const imageRef = useRef()
    const fileInputRef = useRef()
    const textInputRef = useRef()

    const loadModel = async () => {  //loading the model.json file 
        setIsModelLoading(true)
        try {
            const model = await tf.loadGraphModel('/models/model.json')
            setModel(model)
            setIsModelLoading(false)
            console.log("Model loaded successfully")
        } catch (error) {
            console.log(error)
            setIsModelLoading(false)
        }
    }

    const uploadImage = (e) => {
        const { files } = e.target
        if (files.length > 0) {
            const url = URL.createObjectURL(files[0])
            setImageURL(url)
        } else {
            setImageURL(null)
        }
    }
    
    const identify = async () => {
        textInputRef.current.value = ''
        try {
        const imgData = tf.browser
        .fromPixels(imageRef.current)
        .resizeBilinear([256,256])
        .expandDims()

        const results = await model.predict(imgData).data()
        console.log(results) 

        const top3 = Array.from(results).map((item, i) => {
            return {
              precision: item,
              disName: disease[i],
            }
          })
          .sort((a, b) => b.precision - a.precision)
          .slice(0, 3)

        setResults(top3)
       
       
        } catch (error) {
        console.log('No',error)
        }
       
    }

    const handleOnChange = (e) => {
        setImageURL(e.target.value)
        setResults([])
    }

    const triggerUpload = () => {
        fileInputRef.current.click()
    }

    useEffect(() => {
        loadModel()
    }, [])

    useEffect(() => {
        if (imageURL) {
            setHistory([imageURL, ...history])
        }
    }, [imageURL])

    if (isModelLoading) {
        return <h2>Model Loading...</h2>
    }

    return (
        <div className="App">
            <h1 className='header'>Identification</h1>
            <div className='inputHolder'>
                <input type='file' accept='image/*' capture='camera' className='uploadInput' onChange={uploadImage} ref={fileInputRef} />
                <button className='uploadImage' onClick={triggerUpload}>Upload Image</button>
                <span className='or'>OR</span>
                <input type="text" placeholder='Paster image URL' ref={textInputRef} onChange={handleOnChange} />
            </div>
            <div className="mainWrapper">
                <div className="mainContent">
                    <div className="imageHolder">
                        {imageURL && <img src={imageURL} alt="Upload Preview" crossOrigin="anonymous" ref={imageRef} />}
                    </div>
                </div>
                {imageURL && <button className='button' onClick={identify}>Identify Image</button>}
            </div>
            {results.length > 0 && <div>
                {results[0]?.precision*100>50 && <p>{`${results[0]?.disName} -- ${(results[0]?.precision*100).toFixed(3)}%`}</p>}
                <hr/>
                <h4>All Results</h4>
                {results.map((item,i)=>(
                    <p key={i}>{`${item.disName} -- ${(item.precision*100).toFixed(3)}%`}</p>
                ))}
                </div>}
            {history.length > 0 && <div className="recentPredictions">
                <h2>Recent Images</h2>
                <div className="recentImages">
                    {history.map((image, index) => {
                        return (
                            <div className="recentPrediction" key={`${image}${index}`}>
                                <img src={image} alt='Recent Prediction' onClick={() => setImageURL(image)} />
                            </div>
                        )
                    })}
                </div>
            </div>}
        </div>
    );
}

export default App;
