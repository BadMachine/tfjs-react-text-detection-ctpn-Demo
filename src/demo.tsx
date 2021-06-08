import React from 'react';
import * as tf from '@tensorflow/tfjs';
import CTPN, {configInterface} from './main';
tf.ENV.set('WEBGL_PACK', false)

export default class Demo extends React.Component{
	//private _canvasRef: any;// = React.createRef<HTMLCanvasElement | null>();
	private _canvasRef = React.createRef<HTMLCanvasElement | null | undefined>();
	//private _model: Promise<GraphModel>;
	private _canvas: HTMLCanvasElement | undefined;
	private _cfg: configInterface;
	private _model: CTPN;

	constructor(props: any) {
		super(props);
		this._cfg = {
			NMS_FUNCTION: 'TF',
			ANCHOR_SCALES: [16],
			PIXEL_MEANS: tf.tensor([[[102.9801, 115.9465, 122.7717]]]),
			SCALES: [600,] ,
			MAX_SIZE:  1000,
			HAS_RPN: true,
			DETECT_MODE: 'O',
			pre_nms_topN: 12000,
			post_nms_topN: 2000,
			nms_thresh:0.7,
			min_size: 8,
		};
		//this._model = tf.loadGraphModel('https://cdn.jsdelivr.net/gh/BadMachine/tfjs-text-detection-ctpn/ctpn_web/model.json');
		this._model = new CTPN(this._cfg);
	    //this.canvas = React.createRef<HTMLCanvasElement | null>();
	}
	componentDidMount() {
		this._canvas = this._canvasRef.current as HTMLCanvasElement;
	}



	onUpload(event: any){
		const reader = new FileReader();
		const file = event.target.files[0];
		if (!file.type.match('image.*')){
			console.error('not an image');
			return;
		}
		reader.readAsDataURL(file);
		const canvas = this._canvas;

		reader.onload = (fileEvent)=> {

			const ctx = canvas!.getContext('2d');
			const img = new Image();
			//img.src = event.target!.result as string;
			img.src = fileEvent.target!.result as string;
			img.onload = async () => {
				if (img.complete) {
					canvas!.width = img.width;
					canvas!.height = img.height;
					const [predictions, scale] = await this._model.predict(img);
					ctx!.drawImage(img, 0, 0);
					await this._model.draw(canvas as HTMLCanvasElement, predictions as tf.Tensor, scale, 'red');
				}
			};
		};
	}

	render(){
		return(
			<>
			<h1> Text detection CTPN-React demo </h1>
			<div className={'viewPort'} style={{minHeight: '300px', minWidth: '400px'}}>
				{/*@ts-ignore*/}
			<canvas ref={this._canvasRef}/>
			</div>
			<input type="file" onChange={(event:React.FormEvent<HTMLInputElement>) => this.onUpload.apply(this, [event])} />

			</>
		);
	}
}
