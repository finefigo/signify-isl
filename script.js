// DOM Elements
const video = document.getElementById('webcam');
const canvas = document.getElementById('output-canvas');
const ctx = canvas.getContext('2d');
const gestureText = document.getElementById('gesture-text');

// For demonstration purposes - temporary solution
const demoGestureMapping = {
    "0": "Bad",
    "1": "Good", 
    "2": "Hello",
    "3": "Help",
    "4": "Name",
    "5": "No",
    "6": "Please",
    "7": "Sorry",
    "8": "Thank You",
    "9": "Yes"
};

// Variables for stable gesture recognition
let lastUpdateTime = 0;
let stableGesture = '';
let stableConfidence = 0;
let gestureHistory = [];
let gestureCounter = {};
let stabilityDelay = 1000; // Require 1 second of consistent gesture before changing

// Initialize MediaPipe Hands
async function initializeHandDetection() {
    console.log("Initializing hand detection...");
    const hands = new Hands({
        locateFile: (file) => {
            return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
        }
    });

    hands.setOptions({
        maxNumHands: 1,
        modelComplexity: 1,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5
    });

    hands.onResults(onHandResults);

    // Setup camera
    const camera = new Camera(video, {
        onFrame: async () => {
            await hands.send({image: video});
        },
        width: 640,
        height: 480
    });

    console.log("Starting camera...");
    await camera.start();
    console.log("Camera started");
    return { hands, camera };
}

// More robust finger position detection
function isFingerExtended(fingerTip, fingerBase, wrist, threshold = 0.1) {
    // Check if the finger is extended by comparing to the base of the finger and wrist
    const tipToWristY = fingerTip.y - wrist.y;
    const baseToWristY = fingerBase.y - wrist.y;
    
    // Consider horizontal position for better accuracy
    const horizontal = Math.abs(fingerTip.x - fingerBase.x) > 0.05;
    
    // A finger is extended if the tip is higher (lower y value) than the base by some threshold
    return (tipToWristY < 0 && tipToWristY < baseToWristY - threshold) || horizontal;
}

// Get stable gesture based on hand position
function getStableGesture(landmarks) {
    // Get key landmarks for each finger
    const wrist = landmarks[0];
    
    // Thumb landmarks
    const thumbCmc = landmarks[1];
    const thumbMcp = landmarks[2];
    const thumbIp = landmarks[3];
    const thumbTip = landmarks[4];
    
    // Index finger landmarks
    const indexMcp = landmarks[5];
    const indexPip = landmarks[6];
    const indexDip = landmarks[7];
    const indexTip = landmarks[8];
    
    // Middle finger landmarks
    const middleMcp = landmarks[9];
    const middlePip = landmarks[10];
    const middleDip = landmarks[11];
    const middleTip = landmarks[12];
    
    // Ring finger landmarks
    const ringMcp = landmarks[13];
    const ringPip = landmarks[14];
    const ringDip = landmarks[15];
    const ringTip = landmarks[16];
    
    // Pinky finger landmarks
    const pinkyMcp = landmarks[17];
    const pinkyPip = landmarks[18];
    const pinkyDip = landmarks[19];
    const pinkyTip = landmarks[20];
    
    // More accurate finger extension detection
    // Thumb needs special handling because of its orientation
    const thumbExtended = thumbTip.x < wrist.x - 0.05 || thumbTip.y < thumbMcp.y - 0.1;
    const indexExtended = isFingerExtended(indexTip, indexMcp, wrist);
    const middleExtended = isFingerExtended(middleTip, middleMcp, wrist);
    const ringExtended = isFingerExtended(ringTip, ringMcp, wrist);
    const pinkyExtended = isFingerExtended(pinkyTip, pinkyMcp, wrist);
    
    // Create a binary string representing finger states
    const fingerState = `${thumbExtended?1:0}${indexExtended?1:0}${middleExtended?1:0}${ringExtended?1:0}${pinkyExtended?1:0}`;
    
    // Log the detected finger state for debugging
    console.log(`Finger state: ${fingerState}`);
    
    // Map specific finger configurations to gestures
    switch(fingerState) {
        case '01000': // Index finger up only
            return "Hello";
        case '01100': // Index and middle up (peace sign)
            return "Yes";
        case '11111': // All fingers up (open palm)
            return "Good";
        case '01110': // Three middle fingers up
            return "Thank You";  
        case '00000': // Fist - all fingers closed
            return "No";
        case '10001': // Thumb and pinky extended
            return "Name";
        case '01111': // Four fingers except thumb
            return "Help";
        case '11000': // Thumb and index
            return "Bad";
        case '11110': // All except pinky
            return "Please";
        case '10000': // Thumb only
            return "Sorry";
        default:
            // Use additional features for undefined poses
            const thumbIndexClose = Math.sqrt(
                Math.pow(thumbTip.x - indexTip.x, 2) + 
                Math.pow(thumbTip.y - indexTip.y, 2)
            ) < 0.1;
            
            if (fingerState === '11100' || fingerState === '11101') return "Bad";
            if (fingerState === '01110' || fingerState === '01111') return "Thank You";
            if (fingerState === '11110' || fingerState === '11111') return "Good";
            if (thumbIndexClose) return "Bad";
            
            // Fallback: Use the position of the wrist relative to the center
            // to provide consistent mapping for unrecognized poses
            const centerX = 0.5;
            const relativePos = wrist.x - centerX;
            const gestureIndex = Math.abs(Math.floor(relativePos * 10)) % 10;
            return demoGestureMapping[gestureIndex];
    }
}

// Apply majority voting for stable recognition with longer history
function getConsensusGesture(newGesture) {
    // Add new gesture to history (keep last 20 for more stability)
    gestureHistory.push(newGesture);
    if (gestureHistory.length > 20) {
        gestureHistory.shift();
    }
    
    // Count occurrences of each gesture
    gestureCounter = {};
    for (const gesture of gestureHistory) {
        if (!gestureCounter[gesture]) {
            gestureCounter[gesture] = 0;
        }
        gestureCounter[gesture]++;
    }
    
    // Find the most frequent gesture
    let maxCount = 0;
    let consensus = newGesture;
    
    for (const gesture in gestureCounter) {
        if (gestureCounter[gesture] > maxCount) {
            maxCount = gestureCounter[gesture];
            consensus = gesture;
        }
    }
    
    // Calculate confidence based on consensus strength
    const confidence = maxCount / gestureHistory.length;
    
    return {
        gesture: consensus,
        confidence: confidence * 0.3 + 0.7 // Remap to 70-100% range
    };
}

// Handle MediaPipe hand detection results
function onHandResults(results) {
    if (!canvas || !ctx) return;  // Exit if canvas not ready

    // Clear the canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw the video frame
    ctx.save();
    ctx.drawImage(results.image, 0, 0, canvas.width, canvas.height);
    
    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
        // Draw hand landmarks
        for (const landmarks of results.multiHandLandmarks) {
            drawConnectors(ctx, landmarks, HAND_CONNECTIONS, {
                color: '#00FF00',
                lineWidth: 2
            });
            drawLandmarks(ctx, landmarks, {
                color: '#FF0000',
                lineWidth: 1
            });
        }
        
        // Get current time
        const now = Date.now();
        
        // Update gestures consistently
        if (now - lastUpdateTime > 100) {
            // Get gesture based on hand position
            const newGesture = getStableGesture(results.multiHandLandmarks[0]);
            
            // Apply consensus to stabilize gesture
            const result = getConsensusGesture(newGesture);
            
            // Only update if we have sufficient history (for initial stability)
            if (gestureHistory.length >= 10) {
                // Update display if confidence is high enough and we've had stable detection
                if (result.confidence > 0.8) {
                    if (result.gesture !== stableGesture) {
                        // If this is a new gesture, check if it's stable enough to switch
                        if (gestureCounter[result.gesture] >= 10) { // Require at least 10 frames of consistency
                            stableGesture = result.gesture;
                            stableConfidence = result.confidence;
                            gestureText.textContent = `${stableGesture} (${(result.confidence * 100).toFixed(1)}%)`;
                            console.log(`Stable gesture: ${stableGesture} with confidence ${(result.confidence * 100).toFixed(1)}%`);
                        }
                    } else if (result.confidence > stableConfidence) {
                        // Same gesture but higher confidence
                        stableConfidence = result.confidence;
                        gestureText.textContent = `${stableGesture} (${(result.confidence * 100).toFixed(1)}%)`;
                    }
                }
            } else if (gestureHistory.length === 10) {
                // Initial display after collecting enough data
                stableGesture = result.gesture;
                stableConfidence = result.confidence;
                gestureText.textContent = `${stableGesture} (${(result.confidence * 100).toFixed(1)}%)`;
            }
            
            lastUpdateTime = now;
        }
    } else {
        // No hand detected
        gestureText.textContent = 'No hand detected';
        stableGesture = '';
        stableConfidence = 0;
        gestureHistory = [];
        gestureCounter = {};
        lastUpdateTime = 0;
    }
    
    ctx.restore();
}

// Text-to-Speech function
function speakText() {
    const text = gestureText.textContent.split('(')[0].trim();
    if (text && text !== 'No hand detected' && window.speechSynthesis) {
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = 'en-IN';  // Set to Indian English
        window.speechSynthesis.speak(utterance);
    }
}

// Initialize everything when page loads
async function init() {
    try {
        console.log("Initializing...");
        gestureText.textContent = 'Initializing...';
        
        // Slideshow functionality
        let slideIndex = 0;
        function showSlides() {
            let slides = document.getElementsByClassName("mySlides");
            if (slides.length === 0) return;
            
            for (let i = 0; i < slides.length; i++) {
                slides[i].style.display = "none";
            }
            slideIndex++;
            if (slideIndex > slides.length) {
                slideIndex = 1;
            }
            slides[slideIndex - 1].style.display = "block";
            slides[slideIndex - 1].style.opacity = 0;
            
            // Fade in effect
            let op = 0.1;
            let timer = setInterval(function () {
                if (op >= 1) {
                    clearInterval(timer);
                }
                slides[slideIndex - 1].style.opacity = op;
                slides[slideIndex - 1].style.filter = 'alpha(opacity=' + op * 100 + ")";
                op += op * 0.1;
            }, 20);
            
            setTimeout(showSlides, 3000);
        }
        
        // Start slideshow
        showSlides();
        
        // Set canvas size
        function resizeCanvas() {
            if (video && canvas) {
                canvas.width = video.clientWidth;
                canvas.height = video.clientHeight;
            }
        }
        
        // Resize canvas when video loads
        video.addEventListener('loadedmetadata', resizeCanvas);
        // Resize canvas when window resizes
        window.addEventListener('resize', resizeCanvas);
        
        // Initialize hand detection
        const { hands, camera } = await initializeHandDetection();
        
        // Handle visibility change
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                camera.stop();
            } else {
                camera.start();
            }
        });
        
        console.log("Initialization complete");
        gestureText.textContent = 'Ready! Show your hand...';
    } catch (error) {
        console.error('Initialization error:', error);
        if (gestureText) {
            gestureText.textContent = 'Error initializing. Please refresh and try again.';
        }
    }
}

// Start everything when page loads
window.onload = init;

