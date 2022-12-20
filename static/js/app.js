let camera_button = document.querySelector("#start-camera");
let video = document.querySelector("#video");
let start_button = document.querySelector("#start-record");
let stop_button = document.querySelector("#stop-record");
let download_link = document.querySelector("#download-video");
let analyze_button = document.querySelector("#analyze");

let camera_stream = null;
let media_recorder = null;
let blobs_recorded = [];

camera_button.addEventListener('click', async function() {
   	try {
    	camera_stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
    }
    catch(error) {
    	alert(error.message);
    	return;
    }

    video.srcObject = camera_stream;
    camera_button.style.display = 'none';
    video.style.display = 'block';
    start_button.style.display = 'block';
    document.querySelector("#verdict-text").textContent = ""
});

start_button.addEventListener('click', function() {
    media_recorder = new MediaRecorder(camera_stream, { mimeType: 'video/webm' });

    media_recorder.addEventListener('dataavailable', function(e) {
    	blobs_recorded.push(e.data);
    });

    media_recorder.addEventListener('stop', function() {
        let blob = new Blob(blobs_recorded, { type: 'video/webm' })
    	let video_local = URL.createObjectURL(blob);
    	download_link.href = video_local;
        let xhr = new XMLHttpRequest();
        xhr.onreadystatechange = function() {
            if (xhr.readyState == 4 && xhr.status == 200) {
                // alert(xhr.responseText);
            }
        }
        xhr.open("POST", "/record_video");
        xhr.setRequestHeader("Content-type", "video/webm");
        xhr.send(blob);

        start_button.style.display = 'block';
        stop_button.style.display = 'none';
        download_link.style.display = 'inline-block';
        analyze_button.style.display = 'block';
    });

    media_recorder.start(1000);

    start_button.style.display = 'none';
    stop_button.style.display = 'block';
});

stop_button.addEventListener('click', function() {
	media_recorder.stop(); 
});

analyze_button.addEventListener('click', function() {
    let content = document.querySelector("#content")
    let loader = document.querySelector("#loading_screen")
    loader.style.display = "flex"
    content.style.display = "none"
    // if (document.querySelector("#file-name").textContent != "" ) {
      let xhr = new XMLHttpRequest();
      xhr.onreadystatechange = function() {
          if (xhr.readyState == 4 && xhr.status == 200) {
              // alert(xhr.responseText);
              window.location.href = "/"
          }
      }
      let formData = new FormData();
      let fileInput = document.getElementById('file-upload');
      formData.append("file", fileInput.files[0]); 
      xhr.open("POST", "/");
      xhr.send(formData);
    // }
})

document.querySelector("#file-upload").onchange = function(){
    document.querySelector("#file-name").textContent = this.files[0].name;
    analyze_button.style.display = 'block';
    document.querySelector("#verdict-text").textContent = ""

}

const rand = function(min, max) {
    return Math.random() * ( max - min ) + min;
  }
  
  let canvas = document.getElementById('canvas');
  let ctx = canvas.getContext('2d');
  
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
  
  window.addEventListener('resize', function() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    ctx = canvas.getContext('2d');
    ctx.globalCompositeOperation = 'lighter';
  });
  let backgroundColors = [ '#000', '#000' ];
  let colors = [
    [ '#000000', "#ffffff" ],
    [ '#808080', '#373737' ], 
    [ '#c5c6d0' ,'#adadc9' ]
  ];
  let count = 70;
  let blur = [ 12, 70 ];
  let radius = [ 1, 120 ];
  
  ctx.clearRect( 0, 0, canvas.width, canvas.height );
  ctx.globalCompositeOperation = 'lighter';
  
  let grd = ctx.createLinearGradient(0, canvas.height, canvas.width, 0);
  grd.addColorStop(0, backgroundColors[0]);
  grd.addColorStop(1, backgroundColors[1]);
  ctx.fillStyle = grd;
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  
  let items = [];
  
  while(count--) {
      let thisRadius = rand( radius[0], radius[1] );
      let thisBlur = rand( blur[0], blur[1] );
      let x = rand( -100, canvas.width + 100 );
      let y = rand( -100, canvas.height + 100 );
      let colorIndex = Math.floor(rand(0, 299) / 100);
      let colorOne = colors[colorIndex][0];
      let colorTwo = colors[colorIndex][1];
      
      ctx.beginPath();
      ctx.filter = `blur(${thisBlur}px)`;
      let grd = ctx.createLinearGradient(x - thisRadius / 2, y - thisRadius / 2, x + thisRadius, y + thisRadius);
    
      grd.addColorStop(0, colorOne);
      grd.addColorStop(1, colorTwo);
      ctx.fillStyle = grd;
      ctx.fill();
      ctx.arc( x, y, thisRadius, 0, Math.PI * 2 );
      ctx.closePath();
      
      let directionX = Math.round(rand(-99, 99) / 100);
      let directionY = Math.round(rand(-99, 99) / 100);
    
      items.push({
        x: x,
        y: y,
        blur: thisBlur,
        radius: thisRadius,
        initialXDirection: directionX,
        initialYDirection: directionY,
        initialBlurDirection: directionX,
        colorOne: colorOne,
        colorTwo: colorTwo,
        gradient: [ x - thisRadius / 2, y - thisRadius / 2, x + thisRadius, y + thisRadius ],
      });
  }
  
  
  function changeCanvas(timestamp) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    let adjX = 2;
    let adjY = 2;
    let adjBlur = 1;
    items.forEach(function(item) {
      
        if(item.x + (item.initialXDirection * adjX) >= canvas.width && item.initialXDirection !== 0 || item.x + (item.initialXDirection * adjX) <= 0 && item.initialXDirection !== 0) {
          item.initialXDirection = item.initialXDirection * -1;
        }
        if(item.y + (item.initialYDirection * adjY) >= canvas.height && item.initialYDirection !== 0 || item.y + (item.initialYDirection * adjY) <= 0 && item.initialYDirection !== 0) {
          item.initialYDirection = item.initialYDirection * -1;
        }
        
        if(item.blur + (item.initialBlurDirection * adjBlur) >= radius[1] && item.initialBlurDirection !== 0 || item.blur + (item.initialBlurDirection * adjBlur) <= radius[0] && item.initialBlurDirection !== 0) {
          item.initialBlurDirection *= -1;
        }
      
        item.x += (item.initialXDirection * adjX);
        item.y += (item.initialYDirection * adjY);
        item.blur += (item.initialBlurDirection * adjBlur);
        ctx.beginPath();
        ctx.filter = `blur(${item.blur}px)`;
        let grd = ctx.createLinearGradient(item.gradient[0], item.gradient[1], item.gradient[2], item.gradient[3]);
        grd.addColorStop(0, item.colorOne);
        grd.addColorStop(1, item.colorTwo);
        ctx.fillStyle = grd;
        ctx.arc( item.x, item.y, item.radius, 0, Math.PI * 2 );
        ctx.fill();
        ctx.closePath();
      
    });
    window.requestAnimationFrame(changeCanvas);
    
  }
  
  window.requestAnimationFrame(changeCanvas);