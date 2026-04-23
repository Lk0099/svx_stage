FROM nginx:1.25-alpine

# Remove default nginx site
RUN rm /etc/nginx/conf.d/default.conf

# Copy site configuration
COPY nginx.conf /etc/nginx/conf.d/smartvision.conf

# Copy dashboard
COPY index.html /usr/share/nginx/html/index.html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
