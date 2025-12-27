with st.sidebar.expander("🔐 KEYS"):
    tg_token = st.text_input("Bot Token", value=SecretsManager.get("TELEGRAM_TOKEN"), type="password")
    tg_chat = st.text_input("Chat ID", value=SecretsManager.get("TELEGRAM_CHAT_ID"))
    ai_key = st.text_input("AI Key", value=SecretsManager.get("OPENAI_API_KEY"), type="password")

# -------------------------------------------------------------------------
# MODE 1: TITAN MOBILE (Binance)
# -------------------------------------------------------------------------
if mode == "TITAN MOBILE (Crypto)":
    st.sidebar.subheader("📡 BINANCE FEED")
    bases = TitanEngine.get_binance_bases()
    idx = bases.index("BTC") if "BTC" in bases else 0
    base = st.sidebar.selectbox("Asset", bases, index=idx)
    ticker = f"{base}USDT"
    
    c1, c2 = st.sidebar.columns(2)
    with c1: timeframe = st.selectbox("TF", ["15m", "1h", "4h", "1d"], index=1)
    with c2: limit = st.slider("Depth", 100, 500, 200, 50)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🧠 LOGIC")
    amp = st.sidebar.number_input("Amplitude", 2, 100, 10)
    dev = st.sidebar.number_input("Deviation", 0.5, 5.0, 3.0)
    hma_len = st.sidebar.number_input("HMA Len", 10, 200, 50)
    gann_len = st.sidebar.number_input("Gann Len", 2, 50, 3)

    # MAIN UI TITAN
    st.title(f"💠 TITAN: {base}")
    Visuals.render_titan_clock()
    Visuals.render_titan_tape(ticker)
    
    with st.spinner("Connecting to Binance..."):
        df = TitanEngine.get_klines(ticker, timeframe, limit)
    
    if not df.empty:
        df, zones = TitanEngine.run_engine(df, int(amp), dev, int(hma_len), int(gann_len), 55, 1.5, 10)
        last = df.iloc[-1]
        
        # METRICS
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("TREND", "BULL 🟢" if last['is_bull'] else "BEAR 🔴")
        c2.metric("FLUX", f"{last['Apex_Flux']:.2f}")
        c3.metric("STOP", f"{last['entry_stop']:.2f}")
        c4.metric("TP3", f"{last['tp3']:.2f}")
        
        # HTML REPORT
        fg = TitanEngine.calculate_fear_greed(df)
        spec = TitanEngine.detect_special_setups(df)
        st.markdown(TitanEngine.generate_mobile_report(last, fg, spec), unsafe_allow_html=True)
        
        # TELEGRAM
        if st.button("📢 SEND SIGNAL"):
            msg = f"🚀 *TITAN SIGNAL* 🚀\nSymbol: {ticker}\nSide: {'LONG' if last['is_bull'] else 'SHORT'}\nEntry: {last['close']}\nStop: {last['entry_stop']}\nTP1: {last['tp1']}\nTP2: {last['tp2']}\nTP3: {last['tp3']}"
            if send_telegram(tg_token, tg_chat, msg): st.success("SENT")
            else: st.error("FAIL")
        
        # CHART
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
        fig.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['hma'], line=dict(color='#00F0FF', width=1), name='HMA'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['apex_trail'], line=dict(color='orange', width=1, dash='dot'), name='Trail'), row=1, col=1)
        for z in zones: fig.add_shape(type="rect", x0=z['x0'], x1=z['x1'], y0=z['y0'], y1=z['y1'], fillcolor=z['color'], line_width=0, row=1, col=1)
        colors = np.where(df['Apex_Flux'] > 0, '#00E676', '#FF1744')
        fig.add_trace(go.Bar(x=df['timestamp'], y=df['Apex_Flux'], marker_color=colors, name='Flux'), row=2, col=1)
        fig.update_layout(height=500, template='plotly_dark', margin=dict(l=0,r=0,t=10,b=0), hovermode="x unified", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------------------
# MODE 2: AXIOM QUANT (Stocks/YFinance)
# -------------------------------------------------------------------------
else:
    st.sidebar.subheader("📡 MARKET DATA")
    ac_list = AxiomEngine.get_asset_classes()
    ac = st.sidebar.selectbox("Sector", ac_list)
    ticks = AxiomEngine.get_tickers_by_class(ac)
    ticker = st.sidebar.selectbox("Ticker", ticks)
    tf = st.sidebar.selectbox("TF", ["15m", "1h", "4h", "1d", "1wk"], index=3)
    
    st.title(f"💠 AXIOM: {ticker}")
    Visuals.render_axiom_clock()
    Visuals.render_axiom_banner()
    
    with st.spinner("Crunching Physics..."):
        df = AxiomEngine.fetch_data(ticker, tf)
    
    if not df.empty:
        # RUN AXIOM LOGIC
        df = AxiomEngine.calc_chedo(df)
        df = AxiomEngine.calc_rqzo(df)
        df = AxiomEngine.calc_apex_flux(df)
        df = AxiomEngine.calc_smc(df)
        last = df.iloc[-1]
        fund = AxiomEngine.get_fundamentals(ticker)
        macro_p, macro_c = AxiomEngine.get_macro_data()
        
        # DASHBOARD METRICS
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("PRICE", f"{last['Close']:.2f}")
        c2.metric("ENTROPY", f"{last['CHEDO']:.2f}", delta="Risk" if abs(last['CHEDO'])>0.8 else "Stable")
        c3.metric("FLUX", f"{last['Apex_Flux']:.2f}", delta=last['Apex_State'])
        c4.metric("TREND", "BULL" if last['Trend_Dir']==1 else "BEAR")

        # TABS - MODIFIED: ADDED SIGNAL TAB
        tabs = st.tabs(["📢 SIGNAL", "📉 TECH", "🌍 MACRO", "📅 DNA", "🧠 AI", "📊 VOL", "🔮 SIM"])
        
        # SIGNAL TAB (NEW)
        with tabs[0]:
            st.markdown(AxiomEngine.generate_signal_report(df, (macro_p, macro_c), ticker, tf), unsafe_allow_html=True)
            if st.button("📢 SEND SIGNAL TO TELEGRAM", key="axiom_send"):
                msg = AxiomEngine.format_telegram_message(df, (macro_p, macro_c), ticker, tf)
                if send_telegram(tg_token, tg_chat, msg): 
                    st.success("✅ SIGNAL SENT TO TELEGRAM")
                else: 
                    st.error("❌ FAILED TO SEND SIGNAL")
        
        with tabs[1]: # TECH CHART (ORIGINAL)
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.5, 0.25, 0.25], vertical_spacing=0.02)
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['HMA_Trend'], line=dict(color='#fff', width=1, dash='dot'), name='HMA'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['CHEDO'], line=dict(color='#00F0FF', width=2), fill='tozeroy', fillcolor='rgba(0,240,255,0.1)', name='Entropy'), row=2, col=1)
            colors = np.where(df['Apex_Flux'] > 0.6, '#00E676', np.where(df['Apex_Flux'] < -0.6, '#FF1744', '#2979FF'))
            fig.add_trace(go.Bar(x=df.index, y=df['Apex_Flux'], marker_color=colors, name='Flux'), row=3, col=1)
            fig.update_layout(height=700, template='plotly_dark', margin=dict(l=0,r=0,t=10,b=0), hovermode="x unified", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

        with tabs[2]: # MACRO (ORIGINAL)
            c1, c2 = st.columns(2)
            c1.metric("S&P 500", f"${macro_p.get('S&P 500',0):.2f}", f"{macro_c.get('S&P 500',0):.2f}%")
            c2.metric("VIX", f"{macro_p.get('VIX',0):.2f}", f"{macro_c.get('VIX',0):.2f}%")
            if fund: st.write(f"**Fundamentals**: Cap {fund['Market Cap']} | PE {fund['P/E Ratio']}")

        with tabs[3]: # DNA (ORIGINAL)
            dna = AxiomEngine.calc_day_of_week_dna(ticker)
            if dna is not None: st.bar_chart(dna)

        with tabs[4]: # AI (ORIGINAL)
            if st.button("RUN INTELLIGENCE"):
                res = AxiomEngine.analyze_ai(ticker, last['Close'], last['CHEDO'], last['RQZO'], last['Apex_Flux'], ai_key)
                st.info(res)

        with tabs[5]: # VOLUME (ORIGINAL)
            vp, poc = AxiomEngine.calc_volume_profile(df)
            st.bar_chart(vp.set_index('Price')['Volume'])
            st.caption(f"POC: {poc:.2f}")

        with tabs[6]: # MONTE CARLO (ORIGINAL)
            mc = AxiomEngine.run_monte_carlo(df)
            st.line_chart(mc[:, :20])
