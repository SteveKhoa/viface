import "./index.css";
import LoginPage from "./PageLogin";
import WaitingPage from "./PageWaiting";
import SuccessPage from "./PageSuccess";
import FailedPage from "./PageFailed";
import { useState } from "react";
import {
    STATE_LOGIN,
    STATE_WAITING,
    STATE_FAILED,
    STATE_SUCCESS,
} from "./config";

function App() {
    const [state, setState] = useState(STATE_LOGIN);

    return (
        <>
            {state == STATE_LOGIN ? <LoginPage setState={setState} /> : <></>}
            {state == STATE_WAITING ? (
                <WaitingPage setState={setState} />
            ) : (
                <></>
            )}
            {state == STATE_SUCCESS ? (
                <SuccessPage setState={setState} />
            ) : (
                <></>
            )}
            {state == STATE_FAILED ? <FailedPage setState={setState} /> : <></>}
        </>
    );
}

export default App;
