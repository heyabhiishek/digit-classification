const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const predictBtn = document.getElementById("predictBtn");
const clearBtn = document.getElementById("clearBtn");

const prediction = document.getElementById("prediction");
const confidence = document.getElementById("confidence");

let drawing = false;

// Tracks whether the user has actually scratched/drawn
let hasDrawing = false;


// --------------------------------------------------
// CANVAS SETUP
// --------------------------------------------------

function clearCanvas() {

    ctx.fillStyle = "black";
    ctx.fillRect(
        0,
        0,
        canvas.width,
        canvas.height
    );

    ctx.strokeStyle = "white";

    // Similar to handwriting thickness
    ctx.lineWidth = 20;

    ctx.lineCap = "round";
    ctx.lineJoin = "round";
}

clearCanvas();

// Predict disabled initially
predictBtn.disabled = true;


// --------------------------------------------------
// GET MOUSE / TOUCH POSITION
// --------------------------------------------------

function getPosition(event) {

    const rect = canvas.getBoundingClientRect();

    return {
        x:
            (event.clientX - rect.left) *
            (canvas.width / rect.width),

        y:
            (event.clientY - rect.top) *
            (canvas.height / rect.height)
    };
}


// --------------------------------------------------
// START DRAWING
// --------------------------------------------------

canvas.addEventListener(
    "pointerdown",
    (event) => {

        drawing = true;

        canvas.setPointerCapture(
            event.pointerId
        );

        const pos = getPosition(event);

        ctx.beginPath();

        ctx.moveTo(
            pos.x,
            pos.y
        );
    }
);


// --------------------------------------------------
// DRAW
// --------------------------------------------------

canvas.addEventListener(
    "pointermove",
    (event) => {

        if (!drawing) return;

        const pos = getPosition(event);

        ctx.lineTo(
            pos.x,
            pos.y
        );

        ctx.stroke();

        // User has actually moved/scratched on canvas
        hasDrawing = true;

        // Enable Predict only after actual drawing
        predictBtn.disabled = false;
    }
);


// --------------------------------------------------
// STOP DRAWING
// --------------------------------------------------

canvas.addEventListener(
    "pointerup",
    () => {

        drawing = false;

        ctx.closePath();
    }
);


canvas.addEventListener(
    "pointercancel",
    () => {

        drawing = false;
    }
);


// --------------------------------------------------
// CLEAR BUTTON
// --------------------------------------------------

clearBtn.addEventListener(
    "click",
    () => {

        clearCanvas();

        hasDrawing = false;

        // Disable Predict because canvas is empty
        predictBtn.disabled = true;

        prediction.classList.remove("loading");

        prediction.textContent = "Ready";

        confidence.textContent =
            "Draw a digit and click Predict";
    }
);


// --------------------------------------------------
// PREDICT
// --------------------------------------------------

predictBtn.addEventListener(
    "click",
    async () => {

        // Never make a request if user hasn't scratched
        if (!hasDrawing) {
            return;
        }

        // Prevent multiple clicks while analyzing
        predictBtn.disabled = true;

        // Show loading spinner
        prediction.classList.add("loading");

        confidence.textContent =
            "Analyzing your digit...";

        canvas.toBlob(
            async (blob) => {

                const formData =
                    new FormData();

                formData.append(
                    "file",
                    blob,
                    "digit.png"
                );

                try {

                    const response =
                        await fetch(
                            "/predict",
                            {
                                method: "POST",
                                body: formData
                            }
                        );

                    const data =
                        await response.json();

                    // Remove loading state
                    prediction.classList.remove("loading");

                    if (data.error) {

                        prediction.textContent =
                            "!";

                        confidence.textContent =
                            data.error;

                        // Drawing still exists, allow retry
                        predictBtn.disabled = false;

                        return;
                    }

                    prediction.textContent =
                        data.digit;

                    confidence.textContent =
                        `Confidence: ${data.confidence}%`;

                    console.log(
                        "Prediction probabilities:",
                        data.probabilities
                    );

                    // Drawing still exists, allow prediction again
                    predictBtn.disabled = false;

                } catch (error) {

                    console.error(error);

                    // Remove loading state
                    prediction.classList.remove("loading");

                    prediction.textContent =
                        "!";

                    confidence.textContent =
                        "Could not connect to server.";

                    // Allow retry
                    predictBtn.disabled = false;
                }
            },
            "image/png"
        );
    }
);
