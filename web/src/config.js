export const STATE_LOGIN = "state_login";
export const STATE_WAITING = "state_waiting";
export const STATE_SUCCESS = "state_success";
export const STATE_FAILED = "state_failed";

export const ACCESS_TOKEN_ENDPOINT = "http://localhost:8080/token";
export const REGISTER_ENDPOINT = "http://localhost:8080/register";
export const RESOURCE_ENDPOINT = "http://localhost:8080/resource";

console.log(import.meta.env.VITE_WEB_TARGET_PAGE)
export const WEB_TARGET_PAGE = import.meta.env.VITE_WEB_TARGET_PAGE