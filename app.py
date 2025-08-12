import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import json
from urllib.parse import urlparse
import os

# Page configuration
st.set_page_config(page_title="CrUX TTFB Visualizer",
                   page_icon="📊",
                   layout="wide")

st.title("📊 Chrome UX Report Visualizer")
st.markdown(
    "Visualize performance data using Google's CrUX History API")

# Sidebar for inputs
st.sidebar.header("Configuration")

# API Token input
api_token = st.sidebar.text_input(
    "Google Cloud API Key",
    type="password",
    help="Enter your Google Cloud API key with Chrome UX Report API access",
    value=os.getenv("CRUX_API_KEY", ""))

# Analysis mode selection
analysis_mode = st.sidebar.radio(
    "Analysis Mode", ["Single Domain | URL", "Multiple Domains | URLs"],
    help="Choose between analyzing one domain or URL or comparing multiple domains | URLs")

if analysis_mode == "Single Domain | URL":
    # Single URL input
    url_input = st.sidebar.text_input(
        "Website URL or Origin",
        placeholder="https://example.com",
        help="Enter the full URL or origin to analyze")
    domains_list = [url_input] if url_input else []
else:
    # Multiple domains input
    domains_text = st.sidebar.text_area(
        "Website URLs or Origins (one per line)",
        placeholder=
        "https://example.com\nhttps://google.com\nhttps://github.com",
        help="Enter up to 20 URLs or origins, one per line",
        height=150)

    # Process domains list
    if domains_text:
        domains_list = [
            url.strip() for url in domains_text.split('\n') if url.strip()
        ]
        if len(domains_list) > 20:
            st.sidebar.warning(
                f"⚠️ Only the first 20 domains will be analyzed (you entered {len(domains_list)})"
            )
            domains_list = domains_list[:20]
        st.sidebar.info(f"📊 {len(domains_list)} domain(s) ready for analysis")
    else:
        domains_list = []

# Analysis options
show_all_devices = st.sidebar.checkbox(
    "Show all device types",
    value=True,
    help="Display graphs for Phone, Desktop, and Tablet devices")

# Form factor selection (only shown if not showing all devices)
if not show_all_devices:
    form_factor = st.sidebar.selectbox(
        "Device Type", ["PHONE", "DESKTOP", "TABLET"],
        help="Select the device type for analysis")
else:
    form_factor = None

# Data type selection
data_type = st.sidebar.radio(
    "Analysis Level", ["Origin", "URL"],
    help="Origin: All pages under the domain | URL: Specific page only")

# Metrics selection
available_metrics = {
    "experimental_time_to_first_byte": "TTFB (Time to First Byte)",
    "interaction_to_next_paint": "INP (Interaction to Next Paint)",
    "largest_contentful_paint": "LCP (Largest Contentful Paint)",
    "first_contentful_paint": "FCP (First Contentful Paint)"
}

selected_metrics = st.sidebar.multiselect(
    "Performance Metrics",
    options=list(available_metrics.keys()),
    default=["experimental_time_to_first_byte"],
    format_func=lambda x: available_metrics[x],
    help="Select which performance metrics to analyze")

if not selected_metrics:
    st.sidebar.error("⚠️ Please select at least one metric")

# Collection period count
collection_periods = st.sidebar.slider(
    "Collection Periods",
    min_value=1,
    max_value=40,
    value=25,
    help="Number of weekly collection periods to retrieve (max 40 = ~6 months)"
)


def validate_url(url):
    """Validate URL format"""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False


def format_date(date_obj):
    """Format date object to readable string"""
    return f"{date_obj['year']}-{date_obj['month']:02d}-{date_obj['day']:02d}"


def query_crux_history_api(api_key, url, form_factor, data_type, periods,
                           metrics):
    """Query the CrUX History API"""
    endpoint = f"https://chromeuxreport.googleapis.com/v1/records:queryHistoryRecord?key={api_key}"

    # Prepare request body
    request_body = {
        "formFactor": form_factor,
        "metrics": metrics,
        "collectionPeriodCount": periods
    }

    # Add URL or origin based on selection
    if data_type == "Origin":
        request_body["origin"] = url
    else:
        request_body["url"] = url

    try:
        response = requests.post(endpoint,
                                 headers={
                                     'Accept': 'application/json',
                                     'Content-Type': 'application/json'
                                 },
                                 json=request_body,
                                 timeout=30)

        if response.status_code == 200:
            return response.json(), None
        elif response.status_code == 400:
            error_data = response.json()
            return None, f"Bad Request: {error_data.get('error', {}).get('message', 'Invalid request parameters')}"
        elif response.status_code == 403:
            return None, "API Key Error: Invalid or unauthorized API key. Please check your Google Cloud API key."
        elif response.status_code == 404:
            return None, f"No Data Available: The specified {data_type.lower()} '{url}' was not found in the Chrome UX Report dataset."
        else:
            return None, f"API Error: HTTP {response.status_code} - {response.text}"

    except requests.exceptions.Timeout:
        return None, "Request Timeout: The API request timed out. Please try again."
    except requests.exceptions.ConnectionError:
        return None, "Connection Error: Unable to connect to the CrUX API. Please check your internet connection."
    except Exception as e:
        return None, f"Unexpected Error: {str(e)}"


def process_crux_data(data, selected_metrics):
    """Process CrUX API response data for multiple metrics"""
    try:
        record = data.get('record', {})
        collection_periods = record.get('collectionPeriods', [])
        api_metrics = record.get('metrics', {})

        if not api_metrics:
            return None, "No metrics data found in the response"

        # Create base DataFrame with dates
        dates = []
        for period in collection_periods:
            date_str = format_date(period['lastDate'])
            dates.append(pd.to_datetime(date_str))

        if not dates:
            return None, "No valid collection periods found"

        df = pd.DataFrame({'date': dates})

        # Process each selected metric
        metrics_data = {}
        for metric_key in selected_metrics:
            metric_data = api_metrics.get(metric_key, {})

            if metric_data:
                # Extract percentile data (p75)
                percentiles_ts = metric_data.get('percentilesTimeseries', {})
                p75_values = percentiles_ts.get('p75s', [])

                # Create column name
                metric_name = available_metrics.get(
                    metric_key, metric_key).split('(')[0].strip()
                column_name = f"{metric_name}_p75"

                # Add data to dataframe
                values = []
                for i, p75_val in enumerate(p75_values):
                    if i < len(dates) and p75_val is not None:
                        values.append(p75_val)
                    else:
                        values.append(None)

                df[column_name] = values

                # Store histogram data for additional context
                histogram_data = metric_data.get('histogramTimeseries', [])
                metrics_data[metric_key] = {
                    'histogram': histogram_data,
                    'column': column_name
                }

        # Remove rows where all metric values are None
        metric_columns = [col for col in df.columns if col != 'date']
        df = df.dropna(subset=metric_columns, how='all')

        if df.empty:
            return None, "No valid data points found for any selected metric"

        return df, metrics_data

    except Exception as e:
        return None, f"Data processing error: {str(e)}"


def create_metrics_chart(df, url, form_factor, data_type, selected_metrics):
    """Create interactive metrics time series chart"""
    fig = go.Figure()

    # Color palette for different metrics
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    metric_columns = [col for col in df.columns if col != 'date']

    for i, column in enumerate(metric_columns):
        if df[column].notna().any():  # Only add traces with data
            color = colors[i % len(colors)]
            metric_name = column.replace('_p75', '')

            fig.add_trace(
                go.Scatter(
                    x=df['date'],
                    y=df[column],
                    mode='lines+markers',
                    name=f'{metric_name} p75',
                    line=dict(color=color, width=3),
                    marker=dict(size=8, color=color),
                    hovertemplate=
                    f'<b>Date:</b> %{{x}}<br><b>{metric_name} p75:</b> %{{y}} ms<extra></extra>'
                ))

    # Determine title based on metrics
    if len(metric_columns) == 1:
        metric_name = metric_columns[0].replace('_p75', '')
        title = f'{metric_name} Performance Trend - {data_type}: {url} ({form_factor})'
    else:
        title = f'Multi-Metric Performance Trend - {data_type}: {url} ({form_factor})'

    fig.update_layout(title=title,
                      xaxis_title='Collection Period End Date',
                      yaxis_title='Performance Metrics (ms)',
                      hovermode='closest',
                      height=400,
                      showlegend=True)

    # Add grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

    return fig


def create_combined_chart(data_dict, url, data_type, selected_metrics):
    """Create combined chart with all device types for multiple metrics"""
    fig = go.Figure()

    device_colors = {
        'PHONE': '#1f77b4',
        'DESKTOP': '#ff7f0e',
        'TABLET': '#2ca02c'
    }

    for device, df in data_dict.items():
        if df is not None and not df.empty:
            metric_columns = [col for col in df.columns if col != 'date']

            for i, column in enumerate(metric_columns):
                if df[column].notna().any():
                    metric_name = column.replace('_p75', '')
                    device_color = device_colors[device]

                    # Adjust style for multiple metrics
                    line_style = dict(color=device_color, width=3)
                    if len(metric_columns) > 1 and i > 0:
                        line_style['dash'] = 'dash'

                    trace_name = f'{device} {metric_name}' if len(
                        metric_columns) > 1 else f'{device} {metric_name} p75'

                    fig.add_trace(
                        go.Scatter(
                            x=df['date'],
                            y=df[column],
                            mode='lines+markers',
                            name=trace_name,
                            line=line_style,
                            marker=dict(size=8, color=device_color),
                            hovertemplate=
                            f'<b>Device:</b> {device}<br><b>Metric:</b> {metric_name}<br><b>Date:</b> %{{x}}<br><b>Value:</b> %{{y}} ms<extra></extra>'
                        ))

    # Determine title based on metrics
    if len(selected_metrics) == 1:
        metric_name = available_metrics[selected_metrics[0]].split(
            '(')[0].strip()
        title = f'{metric_name} Performance Comparison - {data_type}: {url}'
    else:
        title = f'Multi-Metric Performance Comparison - {data_type}: {url}'

    fig.update_layout(title=title,
                      xaxis_title='Collection Period End Date',
                      yaxis_title='Performance Metrics (ms)',
                      hovermode='closest',
                      height=500,
                      showlegend=True)

    # Add grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

    return fig


def create_multi_domain_chart(domains_data, device_type, data_type,
                              selected_metrics):
    """Create chart comparing multiple domains for a specific device with multiple metrics"""
    fig = go.Figure()

    # Use a color palette for domains
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
        '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78',
        '#98df8a', '#ff9896', '#c5b0d5', '#c49c94', '#f7b6d3', '#c7c7c7',
        '#dbdb8d', '#9edae5'
    ]

    for i, (domain, df) in enumerate(domains_data.items()):
        if df is not None and not df.empty:
            color = colors[i % len(colors)]
            domain_name = domain.replace('https://', '').replace('http://', '')
            metric_columns = [col for col in df.columns if col != 'date']

            for j, column in enumerate(metric_columns):
                if df[column].notna().any():
                    metric_name = column.replace('_p75', '')

                    # Use different line styles for multiple metrics
                    line_style = dict(color=color, width=2)
                    if len(metric_columns) > 1 and j > 0:
                        line_style['dash'] = 'dash'

                    trace_name = f'{domain_name} {metric_name}' if len(
                        metric_columns) > 1 else domain_name

                    fig.add_trace(
                        go.Scatter(
                            x=df['date'],
                            y=df[column],
                            mode='lines+markers',
                            name=trace_name,
                            line=line_style,
                            marker=dict(size=6, color=color),
                            hovertemplate=
                            f'<b>Domain:</b> {domain_name}<br><b>Metric:</b> {metric_name}<br><b>Date:</b> %{{x}}<br><b>Value:</b> %{{y}} ms<extra></extra>'
                        ))

    # Determine title based on metrics
    if len(selected_metrics) == 1:
        metric_name = available_metrics[selected_metrics[0]].split(
            '(')[0].strip()
        title = f'Multi-Domain {metric_name} Comparison - {device_type} ({data_type})'
    else:
        title = f'Multi-Domain Multi-Metric Comparison - {device_type} ({data_type})'

    fig.update_layout(title=title,
                      xaxis_title='Collection Period End Date',
                      yaxis_title='Performance Metrics (ms)',
                      hovermode='closest',
                      height=600,
                      showlegend=True,
                      legend=dict(orientation="v",
                                  yanchor="top",
                                  y=1,
                                  xanchor="left",
                                  x=1.02))

    # Add grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

    return fig


def analyze_multiple_domains(api_token, domains, device_types, data_type,
                             periods, selected_metrics):
    """Analyze multiple domains and return organized data"""
    all_data = {}

    for domain in domains:
        if not validate_url(domain):
            st.warning(f"⚠️ Skipping invalid URL: {domain}")
            continue

        domain_data = {}

        for device in device_types:
            with st.spinner(f"Fetching {device} data for {domain}..."):
                data, error = query_crux_history_api(api_token, domain, device,
                                                     data_type, periods,
                                                     selected_metrics)

                if error:
                    domain_data[device] = None
                else:
                    df, _ = process_crux_data(data, selected_metrics)
                    domain_data[device] = df if not isinstance(df,
                                                               str) else None

        all_data[domain] = domain_data

    return all_data


def display_summary_stats(df, device_name=None, selected_metrics=None):
    """Display summary statistics for multiple metrics"""
    metric_columns = [col for col in df.columns if col != 'date']

    device_label = f" ({device_name})" if device_name else ""

    for column in metric_columns:
        if df[column].notna().any():
            metric_name = column.replace('_p75', '')

            # Create metric row
            st.markdown(f"**{metric_name}{device_label}**")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                latest_val = df[column].iloc[-1]
                if pd.notna(latest_val):
                    st.metric("Latest", f"{latest_val:.0f} ms")
                else:
                    st.metric("Latest", "N/A")

            with col2:
                avg_val = df[column].mean()
                if pd.notna(avg_val):
                    st.metric("Average", f"{avg_val:.0f} ms")
                else:
                    st.metric("Average", "N/A")

            with col3:
                min_val = df[column].min()
                if pd.notna(min_val):
                    st.metric("Best", f"{min_val:.0f} ms")
                else:
                    st.metric("Best", "N/A")

            with col4:
                max_val = df[column].max()
                if pd.notna(max_val):
                    st.metric("Worst", f"{max_val:.0f} ms")
                else:
                    st.metric("Worst", "N/A")

            st.markdown("---")


def display_multi_device_summary(data_dict):
    """Display summary for all devices"""
    st.subheader("📈 Performance Summary - All Devices")

    for device, df in data_dict.items():
        if df is not None and not df.empty:
            st.markdown(f"**{device}**")
            display_summary_stats(df, device)
            st.markdown("---")


# Main application logic
if st.sidebar.button("🔍 Analyze data", type="primary"):
    # Validation
    if not api_token:
        st.error("❌ Please enter your Google Cloud API key")
        st.stop()

    if not domains_list:
        if analysis_mode == "Single Domain | URL":
            st.error("❌ Please enter a URL to analyze")
        else:
            st.error("❌ Please enter at least one URL to analyze")
        st.stop()

    if not selected_metrics:
        st.error("❌ Please select at least one metric to analyze")
        st.stop()

    # Validate URLs
    valid_domains = []
    for domain in domains_list:
        if validate_url(domain):
            valid_domains.append(domain)
        else:
            st.warning(f"⚠️ Skipping invalid URL: {domain}")

    if not valid_domains:
        st.error("❌ No valid URLs provided")
        st.stop()

    domains_list = valid_domains

    # Determine analysis scope
    if analysis_mode == "Multiple Domains | URLs" and len(domains_list) > 1:
        # Multi-domain analysis
        st.info(f"🔄 Analyzing {len(domains_list)} domains | URLs...")

        if show_all_devices:
            device_types = ["PHONE", "DESKTOP", "TABLET"]
        else:
            device_types = [form_factor]

        # Fetch all data
        all_domains_data = analyze_multiple_domains(api_token, domains_list,
                                                    device_types, data_type,
                                                    collection_periods,
                                                    selected_metrics)

        # Check if we have any valid data
        has_data = False
        for domain_data in all_domains_data.values():
            for device_data in domain_data.values():
                if device_data is not None and not device_data.empty:
                    has_data = True
                    break

        if not has_data:
            st.error("❌ No data available for any domain or device")
            st.stop()

        st.success(f"✅ Analysis complete for {len(domains_list)} domains | URLs!")

        # Create multi-domain comparison charts for each device
        for device in device_types:
            device_domain_data = {}
            for domain, domain_data in all_domains_data.items():
                if domain_data.get(
                        device) is not None and not domain_data[device].empty:
                    device_domain_data[domain] = domain_data[device]

            if device_domain_data:
                st.subheader(f"📊 {device} - Multi-Domain Comparison")
                multi_fig = create_multi_domain_chart(device_domain_data,
                                                      device, data_type,
                                                      selected_metrics)
                st.plotly_chart(multi_fig, use_container_width=True)

                # Summary table for this device
                st.subheader(f"📈 {device} Performance Summary")
                summary_data = []
                for domain, df in device_domain_data.items():
                    domain_name = domain.replace('https://',
                                                 '').replace('http://', '')
                    metric_columns = [
                        col for col in df.columns if col != 'date'
                    ]

                    row_data = {'Domain': domain_name}

                    for column in metric_columns:
                        if df[column].notna().any():
                            metric_name = column.replace('_p75', '')
                            latest_val = df[column].iloc[-1] if pd.notna(
                                df[column].iloc[-1]) else None
                            avg_val = df[column].mean() if df[column].notna(
                            ).any() else None

                            if latest_val is not None:
                                row_data[
                                    f'{metric_name} Latest (ms)'] = f"{latest_val:.0f}"

                                # Performance classification for TTFB only
                                if 'TTFB' in metric_name:
                                    if latest_val <= 800:
                                        row_data['TTFB Status'] = "🟢 Good"
                                    elif latest_val <= 1800:
                                        row_data[
                                            'TTFB Status'] = "🟡 Needs Improvement"
                                    else:
                                        row_data['TTFB Status'] = "🔴 Poor"

                            if avg_val is not None:
                                row_data[
                                    f'{metric_name} Average (ms)'] = f"{avg_val:.0f}"

                    summary_data.append(row_data)

                if summary_data:
                    summary_df = pd.DataFrame(summary_data)
                    st.dataframe(summary_df, use_container_width=True)
                st.markdown("---")

        # Export functionality for multi-domain
        st.subheader("💾 Export Data")
        for device in device_types:
            device_domain_data = {}
            for domain, domain_data in all_domains_data.items():
                if domain_data.get(device) is not None:
                    device_domain_data[domain] = domain_data[device]

            if device_domain_data:
                # Combine all domain data for this device
                combined_data = []
                for domain, df in device_domain_data.items():
                    if df is not None and not df.empty:
                        df_copy = df.copy()
                        df_copy['domain'] = domain.replace('https://',
                                                           '').replace(
                                                               'http://', '')
                        combined_data.append(df_copy)

                if combined_data:
                    combined_df = pd.concat(combined_data, ignore_index=True)
                    csv_data = combined_df.to_csv(index=False)
                    st.download_button(
                        label=f"📥 Download {device} Multi-Domain CSV",
                        data=csv_data,
                        file_name=
                        f"ttfb_multi_domain_{device}_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        key=f"download_multi_{device}")

    elif len(domains_list) == 1:
        # Single domain analysis
        url_input = domains_list[0]

        if show_all_devices:
            # Fetch data for all device types
            device_types = ["PHONE", "DESKTOP", "TABLET"]
            device_data = {}

            for device in device_types:
                with st.spinner(f"Fetching data for {device}..."):
                    data, error = query_crux_history_api(
                        api_token, url_input, device, data_type,
                        collection_periods, selected_metrics)

                    if error:
                        st.warning(f"⚠️ {device}: {error}")
                        device_data[device] = None
                    else:
                        df, metrics_data = process_crux_data(
                            data, selected_metrics)
                        if isinstance(df, str):  # Error message
                            st.warning(f"⚠️ {device}: {df}")
                            device_data[device] = None
                        else:
                            device_data[device] = df

            # Check if we have any data
            valid_data = {
                k: v
                for k, v in device_data.items()
                if v is not None and not v.empty
            }

            if not valid_data:
                st.error("❌ No data available for any device type")
                st.stop()

            st.success(
                f"✅ Data retrieved for {len(valid_data)} device type(s)!")

            # Combined comparison chart
            st.subheader("📊 Performance Comparison")
            combined_fig = create_combined_chart(valid_data, url_input,
                                                 data_type, selected_metrics)
            st.plotly_chart(combined_fig, use_container_width=True)

            # Individual device charts
            st.subheader("📱📺🖥️ Individual Device Performance")

            for device, df in valid_data.items():
                st.markdown(f"### {device}")

                # Summary stats for this device
                display_summary_stats(df, device, selected_metrics)

                # Individual chart
                device_fig = create_metrics_chart(df, url_input, device,
                                                  data_type, selected_metrics)
                st.plotly_chart(device_fig, use_container_width=True)

                # Performance classification for TTFB if present
                ttfb_column = None
                for col in df.columns:
                    if 'TTFB' in col:
                        ttfb_column = col
                        break

                if ttfb_column and df[ttfb_column].notna().any():
                    latest_ttfb = df[ttfb_column].iloc[-1]
                    if pd.notna(latest_ttfb):
                        if latest_ttfb <= 800:
                            st.success(
                                f"🟢 {device}: Good TTFB performance (≤ 800ms)")
                        elif latest_ttfb <= 1800:
                            st.warning(
                                f"🟡 {device}: Needs improvement (800ms - 1800ms)"
                            )
                        else:
                            st.error(
                                f"🔴 {device}: Poor TTFB performance (> 1800ms)"
                            )

                st.markdown("---")

            # Combined raw data
            st.subheader("📋 Raw Data")
            for device, df in valid_data.items():
                with st.expander(f"View {device} Raw Data"):
                    st.dataframe(df.sort_values('date', ascending=False),
                                 use_container_width=True)

            # Export functionality for all devices
            st.subheader("💾 Export Data")
            for device, df in valid_data.items():
                csv_data = df.to_csv(index=False)
                st.download_button(
                    label=f"📥 Download {device} CSV",
                    data=csv_data,
                    file_name=
                    f"ttfb_data_{device}_{url_input.replace('https://', '').replace('http://', '').replace('/', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    key=f"download_{device}")

        else:
            # Single device analysis (original logic)
            with st.spinner(f"Fetching performance data for {url_input}..."):
                data, error = query_crux_history_api(api_token, url_input,
                                                     form_factor, data_type,
                                                     collection_periods,
                                                     selected_metrics)

                if error:
                    st.error(f"❌ {error}")
                    st.stop()

                df, metrics_data = process_crux_data(data, selected_metrics)

                if isinstance(df, str):  # Error message
                    st.error(f"❌ {df}")
                    st.stop()

            st.success("✅ Data retrieved successfully!")

            # Summary statistics
            st.subheader("📈 Performance Summary")
            display_summary_stats(df, selected_metrics=selected_metrics)

            # Time series chart
            st.subheader("📊 Performance Trend Over Time")
            fig = create_metrics_chart(df, url_input, form_factor, data_type,
                                       selected_metrics)
            st.plotly_chart(fig, use_container_width=True)

            # Data insights
            st.subheader("💡 Insights")

            # Trend analysis for each metric
            if df is not None and len(df) >= 10:
                metric_columns = [col for col in df.columns if col != 'date']
                for column in metric_columns:
                    if df[column].notna().sum() >= 10:
                        metric_name = column.replace('_p75', '')
                        recent_trend = df[column].iloc[-5:].mean(
                        ) - df[column].iloc[-10:-5].mean()
                        if recent_trend > 50:
                            st.warning(
                                f"📈 {metric_name} has increased significantly in recent periods"
                            )
                        elif recent_trend < -50:
                            st.success(
                                f"📉 {metric_name} has improved significantly in recent periods"
                            )
                        else:
                            st.info(
                                f"➡️ {metric_name} has remained relatively stable"
                            )

            # Performance classification for TTFB
            if df is not None:
                ttfb_column = None
                for col in df.columns:
                    if 'TTFB' in col:
                        ttfb_column = col
                        break

                if ttfb_column and df[ttfb_column].notna().any():
                    latest_ttfb = df[ttfb_column].iloc[-1]
                    if pd.notna(latest_ttfb):
                        if latest_ttfb <= 800:
                            st.success("🟢 Good TTFB performance (≤ 800ms)")
                        elif latest_ttfb <= 1800:
                            st.warning("🟡 Needs improvement (800ms - 1800ms)")
                        else:
                            st.error("🔴 Poor TTFB performance (> 1800ms)")

                # Raw data table
                with st.expander("📋 View Raw Data"):
                    st.dataframe(df.sort_values('date', ascending=False),
                                 use_container_width=True)

                # Export functionality
                st.subheader("💾 Export Data")
                csv_data = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_data,
                    file_name=
                    f"ttfb_data_{url_input.replace('https://', '').replace('http://', '').replace('/', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv")

# Information section
st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ About")
st.sidebar.markdown("""
**Dev: Jan-Willem Bobbink**

**Analysis Modes:**
- Single domain | URL: Detailed performance analysis
- Multiple domains | URL: Compare up to 20 domains or URLs
- Multi-device support: Phone, Desktop, Tablet
- Multi-metric analysis: TTFB, INP, LCP, FCP
""")

st.sidebar.markdown("---")
st.sidebar.markdown("🔗 **Resources:**")
st.sidebar.markdown(
    "- [CrUX History API Docs](https://developer.chrome.com/docs/crux/history-api)"
)
st.sidebar.markdown(
    "- [Get API Key](https://console.cloud.google.com/apis/credentials)")

# Footer
st.markdown("---")
st.markdown("Built for https://www.notprovided.eu • Data from Chrome UX Report")
