// GameMaker extension that steps the patched game in lockstep with the trainer over TCP.
// Build: x86_64-w64-mingw32-gcc -O2 -shared -o ufo50rl.dll ufo50rl.c -lws2_32 -luser32

#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <stdint.h>
#include <stdlib.h>

#define EXPORT __declspec(dllexport)
#define REPLY_LEN 16

static SOCKET sock = INVALID_SOCKET;
static int hide_window = 0;
static HWND game_window = NULL;

static BOOL CALLBACK find_game_window(HWND w, LPARAM out) {
    DWORD pid = 0;
    GetWindowThreadProcessId(w, &pid);
    if (pid == GetCurrentProcessId() && IsWindowVisible(w)) {
        *(HWND *)out = w;
        return FALSE;
    }
    return TRUE;
}

// Keeps the game window hidden; GameMaker may show it again after creating or resizing it.
static void keep_hidden(void) {
    if (!hide_window) return;
    if (!game_window) EnumWindows(find_game_window, (LPARAM)&game_window);
    if (game_window && IsWindowVisible(game_window)) ShowWindow(game_window, SW_HIDE);
}

static int send_all(const char *p, int n) {
    while (n > 0) {
        int k = send(sock, p, n, 0);
        if (k <= 0) return 0;
        p += k;
        n -= k;
    }
    return 1;
}

static int recv_all(char *p, int n) {
    while (n > 0) {
        int k = recv(sock, p, n, 0);
        if (k <= 0) return 0;
        p += k;
        n -= k;
    }
    return 1;
}

// Returns 1 if UFO50RL_PORT is set and the trainer accepted the connection, else 0.
// The game window is hidden unless UFO50RL_SHOW_WINDOW is set.
EXPORT double rl_init(void) {
    const char *port = getenv("UFO50RL_PORT");
    const char *host = getenv("UFO50RL_HOST");
    if (!port) return 0;
    if (!host) host = "127.0.0.1";
    hide_window = getenv("UFO50RL_SHOW_WINDOW") == NULL;

    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) return 0;

    struct addrinfo hints = {0}, *res = NULL;
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(host, port, &hints, &res) != 0) return 0;

    sock = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (sock == INVALID_SOCKET || connect(sock, res->ai_addr, (int)res->ai_addrlen) != 0) {
        freeaddrinfo(res);
        if (sock != INVALID_SOCKET) closesocket(sock);
        sock = INVALID_SOCKET;
        return 0;
    }
    freeaddrinfo(res);

    int one = 1;
    setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, (const char *)&one, sizeof(one));
    return 1;
}

// buf[0..16) receives the trainer's reply; buf[16..len) is sent as a length-prefixed message.
// Blocks until the reply arrives. Returns 0 once the trainer has disconnected.
EXPORT double rl_step(char *buf, double len) {
    if (sock == INVALID_SOCKET) return 0;
    keep_hidden();
    uint32_t n = (uint32_t)len - REPLY_LEN;
    if (!send_all((const char *)&n, 4) || !send_all(buf + REPLY_LEN, (int)n) || !recv_all(buf, REPLY_LEN)) {
        closesocket(sock);
        sock = INVALID_SOCKET;
        return 0;
    }
    return 1;
}
